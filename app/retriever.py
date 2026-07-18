#python -m app.retriever

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNK_TEXT_KEY = "text"

INDEX_DIR = Path(os.environ.get("VECTOR_DB_PATH", "data/chroma_index"))
COLLECTION_NAME = "approved_docs"

# Chroma's own default index space is squared L2, not cosine -- set this
# explicitly wherever the collection is created so retrieve()'s "1 -
# distance" similarity conversion is actually valid.
COLLECTION_METADATA = {"hnsw:space": "cosine"}
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Texts are embedded/added to Chroma in batches rather than all at once, to
# keep memory bounded for a large index build.
EMBED_BATCH_SIZE = 64

# Per spec 12.2: below this similarity, treat retrieval as a miss.
#
# Calibrated against this project's actual embedding config (OpenAI
# text-embedding-3-small, cosine space), not a generic default: sampled
# top-1 similarity for off-topic questions tops out around 0.15, for
# on-topic-but-genuinely-unsupported questions (e.g. exact pricing, which
# the docs never cover) around 0.27, and for real, answerable questions
# starts around 0.37. 0.32 sits in that real gap. Different embedding
# providers/models have different score distributions -- re-sample before
# reusing this value elsewhere.
NO_MATCH_THRESHOLD = float(os.environ.get("RETRIEVER_NO_MATCH_THRESHOLD", 0.32))

# Per spec 12.2: between NO_MATCH_THRESHOLD and this, answer with low confidence.
#
# The naive read of the sampled gap here (weak/indirect matches topping out
# ~0.45, direct hits on approved content starting ~0.58) suggests anywhere
# in that range works. It doesn't: several catalogues share a near-identical
# generic "Product Benefits" boilerplate paragraph that acts as a similarity
# magnet -- e.g. "How is the leak detector packaged for shipping?" (a
# question the docs don't actually answer) scores 0.5431 against that
# paragraph alone. A threshold below that would label a non-answer "high
# confidence", which is the inflation this must NOT do. 0.56 sits just
# above that confirmed bad match and just below the lowest confirmed direct
# hit (0.5840). Narrow margin -- re-sample if it starts misclassifying.
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("RETRIEVER_LOW_CONFIDENCE_THRESHOLD", 0.56))

_embedder: Optional[SentenceTransformer] = None


class RetrieverError(RuntimeError):
    """Wraps embedding/vector-store failures so callers can catch one
    specific type instead of a raw Chroma/model exception."""


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def get_embedder() -> SentenceTransformer:
    """Loads the embedding model once and reuses it."""
    global _embedder
    if _embedder is None:
        try:
            _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as e:
            raise RetrieverError(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}") from e
    return _embedder


_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RetrieverError(
                "OPENAI_API_KEY is not set -- required when EMBEDDING_PROVIDER=openai."
            )
        try:
            from openai import OpenAI

            _openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            raise RetrieverError(f"Failed to initialize the OpenAI client: {e}") from e
    return _openai_client


def call_openai_embeddings(texts: list[str]) -> list[list[float]]:
    """Embeds one batch of texts via the OpenAI embeddings API."""
    client = _get_openai_client()
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


def _embed_batch(batch: list[str]) -> list[list[float]]:
    if EMBEDDING_PROVIDER == "openai":
        return call_openai_embeddings(batch)
    return get_embedder().encode(batch, show_progress_bar=False).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turns texts into embedding vectors in batches of EMBED_BATCH_SIZE.
    Provider is controlled by EMBEDDING_PROVIDER -- this is the only place
    that decision is made."""
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    try:
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            all_embeddings.extend(_embed_batch(batch))
    except Exception as e:
        raise RetrieverError(f"Embedding failed: {e}") from e

    return all_embeddings


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def get_client() -> "chromadb.PersistentClient":
    try:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(INDEX_DIR))
    except Exception as e:
        raise RetrieverError(f"Failed to open the vector index at {INDEX_DIR}: {e}") from e


def get_collection():
    client = get_client()
    try:
        return client.get_or_create_collection(COLLECTION_NAME, metadata=COLLECTION_METADATA)
    except Exception as e:
        raise RetrieverError(f"Failed to open collection '{COLLECTION_NAME}': {e}") from e


def index_size() -> int:
    """Returns 0 on failure rather than raising, so a status display
    doesn't crash just because the index isn't built yet."""
    try:
        return get_collection().count()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Build / rebuild the index (spec 12.1)
# ---------------------------------------------------------------------------
def _validate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Skips chunks missing a usable CHUNK_TEXT_KEY instead of letting a
    malformed chunk raise mid-build."""
    valid: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        text = chunk.get(CHUNK_TEXT_KEY) if isinstance(chunk, dict) else None
        if not isinstance(text, str) or not text.strip():
            print(
                f"WARNING: skipping chunk at index {i} -- missing or empty "
                f"'{CHUNK_TEXT_KEY}' key: {chunk!r}",
                file=sys.stderr,
            )
            continue
        valid.append(chunk)
    return valid


def _dedupe_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Drops exact-duplicate chunk text (content-based, keeping the first
    occurrence). Returns (deduped_chunks, content_hash_per_chunk) -- the
    hashes double as stable, deterministic Chroma IDs in build_index."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    hashes: list[str] = []
    duplicate_count = 0

    for chunk in chunks:
        text = chunk[CHUNK_TEXT_KEY]
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_hash in seen:
            duplicate_count += 1
            continue
        seen.add(content_hash)
        deduped.append(chunk)
        hashes.append(content_hash)

    if duplicate_count:
        print(f"WARNING: skipped {duplicate_count} duplicate chunk(s) (identical text).", file=sys.stderr)

    return deduped, hashes


def build_index(chunks: list[dict[str, Any]]) -> int:
    """Embeds and stores chunks in the vector database, replacing whatever
    was indexed before. Invalid or duplicate chunks are skipped with a
    warning rather than raising. Returns the number of chunks actually
    indexed (may be less than len(chunks))."""
    if not chunks:
        return 0

    valid_chunks = _validate_chunks(chunks)
    if not valid_chunks:
        return 0

    deduped_chunks, ids = _dedupe_chunks(valid_chunks)
    if not deduped_chunks:
        return 0

    client = get_client()

    # Drop and recreate the collection so chunks from deleted/edited
    # documents don't linger from a previous index build.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # fine if it didn't exist yet (first-ever build)

    try:
        collection = client.get_or_create_collection(COLLECTION_NAME, metadata=COLLECTION_METADATA)
    except Exception as e:
        raise RetrieverError(f"Failed to create collection '{COLLECTION_NAME}': {e}") from e

    texts = [c[CHUNK_TEXT_KEY] for c in deduped_chunks]
    # Chroma rejects None as a metadata value, so drop any key whose value
    # is None (e.g. a docx/pdf chunk's absent slide_number) before sending.
    metadatas = [
        {k: v for k, v in c.items() if k != CHUNK_TEXT_KEY and v is not None}
        for c in deduped_chunks
    ]

    embeddings = embed_texts(texts)  # already batched internally

    try:
        for start in range(0, len(deduped_chunks), EMBED_BATCH_SIZE):
            end = start + EMBED_BATCH_SIZE
            collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )
    except Exception as e:
        raise RetrieverError(f"Failed to add chunks to the vector store: {e}") from e

    return len(deduped_chunks)


# ---------------------------------------------------------------------------
# "List all products" queries (spec 12.2 extension)
# ---------------------------------------------------------------------------
# Top-k nearest-neighbor search answers "which chunks best match this
# text", not "enumerate every product" -- with more distinct products in
# the corpus than top_k, something is always dropped, and which one is
# arbitrary (whichever chunks happen to be worded less similarly to the
# question), not a reflection of which products exist. Detected by keyword
# pattern rather than similarity, consistent with this project's existing
# deterministic, keyword-based classification (see intent.py,
# claim_checker.py) -- "enumerate everything" isn't itself a
# similarity-driven question.
_LIST_ALL_PRODUCTS_PATTERNS = [
    re.compile(r"\ball\b.{0,15}\bproducts?\b", re.IGNORECASE),
    re.compile(r"\bwhat\b.{0,10}\bproducts?\b.{0,15}\b(available|offer|have|sell|make)\b", re.IGNORECASE),
    re.compile(r"\bproducts?\s+(lineup|line-up|catalog|catalogue|range|portfolio)\b", re.IGNORECASE),
    re.compile(r"\blist\b.{0,10}\bproducts?\b", re.IGNORECASE),
]


def _is_list_all_products_query(query: str) -> bool:
    return any(pattern.search(query) for pattern in _LIST_ALL_PRODUCTS_PATTERNS)


def _retrieve_one_per_document(query_embedding: list[list[float]], collection) -> list[dict[str, Any]]:
    """One best-matching chunk per distinct document instead of a single
    global top-k, so every indexed product is represented at least once.
    Reuses the same per-chunk similarity search as retrieve()'s normal
    path, just partitioned by document via Chroma's `where` filter."""
    all_metadata = collection.get(include=["metadatas"])["metadatas"]
    document_names = sorted({m["document_name"] for m in all_metadata})

    matches = []
    for document_name in document_names:
        result = collection.query(
            query_embeddings=query_embedding,
            n_results=1,
            where={"document_name": document_name},
        )
        if not result["documents"][0]:
            continue
        matches.append(
            {
                "text": result["documents"][0][0],
                "metadata": result["metadatas"][0][0],
                "similarity": 1 - result["distances"][0][0],
            }
        )

    matches.sort(key=lambda m: m["metadata"].get("document_name", ""))
    return matches


# ---------------------------------------------------------------------------
# Retrieval (spec 12.2)
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = 5) -> dict[str, Any]:
    """Retrieves the top_k most relevant chunks for a question.

    Per spec 12.2: confidence is "none" (empty matches) if nothing clears
    NO_MATCH_THRESHOLD, "low" if the best match is below
    LOW_CONFIDENCE_THRESHOLD, else "high".

    Returns:
        {"matches": [{"text", "metadata", "similarity"}, ...],
         "confidence": "high" | "low" | "none"}
    """
    collection = get_collection()

    try:
        collection_count = collection.count()
    except Exception as e:
        raise RetrieverError(f"Failed to read the vector store: {e}") from e

    if collection_count == 0:
        return {"matches": [], "confidence": "none"}

    query_embedding = embed_texts([query])

    if _is_list_all_products_query(query):
        try:
            matches = _retrieve_one_per_document(query_embedding, collection)
        except Exception as e:
            raise RetrieverError(f"Vector search failed: {e}") from e
    else:
        try:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, collection_count),
            )
        except Exception as e:
            raise RetrieverError(f"Vector search failed: {e}") from e

        matches = []
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            # Chroma returns cosine distance here (0 = identical); similarity
            # is the complement, higher is better.
            similarity = 1 - distance
            matches.append({"text": text, "metadata": metadata, "similarity": similarity})

    # max() rather than matches[0]: the list-all-products branch above
    # orders matches by document name for a readable enumeration, not by
    # similarity, so the top-ranked match isn't necessarily first anymore.
    best_similarity = max((m["similarity"] for m in matches), default=0.0)

    if not matches or best_similarity < NO_MATCH_THRESHOLD:
        return {"matches": [], "confidence": "none"}
    elif best_similarity < LOW_CONFIDENCE_THRESHOLD:
        confidence = "low"
    else:
        confidence = "high"

    return {"matches": matches, "confidence": confidence}


# ---------------------------------------------------------------------------
# Wiring to document_loader.py + chunker.py
# ---------------------------------------------------------------------------
# Kept in sync by hand with document_loader.py's _EXTRACTORS keys.
SUPPORTED_DOC_EXTENSIONS = {".docx", ".pptx", ".pdf", ".csv", ".md", ".markdown", ".txt"}


def load_and_chunk_approved_docs(docs_dir: str | Path = "data/approved_docs") -> list[dict[str, Any]]:
    """Loads and chunks every supported file in docs_dir, returning plain
    dicts ready for build_index(). A file that fails to load is skipped
    with a warning rather than aborting the whole reindex."""
    from app.chunker import chunk_documents
    from app.document_loader import load_document

    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        print(f"WARNING: {docs_dir} does not exist -- nothing to index.", file=sys.stderr)
        return []

    documents: list[dict[str, Any]] = []
    for path in sorted(docs_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS:
            continue
        try:
            documents.append(load_document(path))
        except Exception as e:
            print(f"WARNING: failed to load {path.name}: {e}", file=sys.stderr)

    if not documents:
        print(f"WARNING: no supported documents found in {docs_dir}.", file=sys.stderr)
        return []

    chunks = chunk_documents(documents)
    return [chunk.to_dict() for chunk in chunks]


# ---------------------------------------------------------------------------
# CLI: rebuild the index, or -- if a question is given as an argument --
# debug-print what retrieve() finds for it, without touching indexing.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = retrieve(question)
        print(f"Confidence: {result['confidence']}")
        for match in result["matches"]:
            metadata = match["metadata"]
            section = metadata.get("section_heading") or metadata.get("page_number") or metadata.get("slide_number") or "N/A"
            print(f"Similarity: {match['similarity']:.4f}")
            print(f"Document: {metadata.get('document_name', 'unknown')}")
            print(f"Section/Page: {section}")
            print(f"Text: {match['text'][:200]}")
            print("-" * 40)
    else:
        docs_dir = Path("data/approved_docs")
        print(f"Loading and chunking documents from {docs_dir} ...")
        chunk_dicts = load_and_chunk_approved_docs(docs_dir)
        print(f"Produced {len(chunk_dicts)} chunks.")

        try:
            count = build_index(chunk_dicts)
            print(f"Indexed {count} chunks from {docs_dir}. Run: python -m app.retriever")
        except RetrieverError as e:
            print(f"Index build failed: {e}", file=sys.stderr)
            raise SystemExit(1)
