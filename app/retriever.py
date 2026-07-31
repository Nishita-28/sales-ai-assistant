#python -m app.retriever
from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import chromadb
from dotenv import load_dotenv

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNK_TEXT_KEY = "text"

INDEX_DIR = Path(os.environ.get("VECTOR_DB_PATH", "data/chroma_index"))
COLLECTION_NAME = "approved_docs"

# Chroma defaults to squared L2 distance, not cosine -- set this explicitly
# so the "1 - distance" similarity conversion below is valid.
COLLECTION_METADATA = {"hnsw:space": "cosine"}
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Batches embedding calls to keep memory bounded for a large index build.
EMBED_BATCH_SIZE = 64

# Below this similarity, treat retrieval as a miss. Calibrated against
# sampled scores for this embedding model -- re-sample if the model changes.
NO_MATCH_THRESHOLD = float(os.environ.get("RETRIEVER_NO_MATCH_THRESHOLD", 0.32))

# Between NO_MATCH_THRESHOLD and this, answer with low confidence. A shared
# boilerplate paragraph across documents can inflate unrelated questions'
# scores, so this sits above that rather than in the middle of the gap.
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("RETRIEVER_LOW_CONFIDENCE_THRESHOLD", 0.56))

# How many extra keyword-matched chunks can be merged into the vector top_k
# per query (see _keyword_boost_matches).
KEYWORD_BOOST_MAX_EXTRA = 3

# Minimum chunks retrieved per product in a comparison question (see
# _retrieve_balanced_across_products) -- higher than a plain top_k split,
# since a short, specific spec row can lose out to similar-sounding
# feature bullets at a narrower count.
COMPARISON_CHUNKS_PER_PRODUCT = 10

_embedder: Optional[SentenceTransformer] = None


class RetrieverError(RuntimeError):
    """Wraps embedding/vector-store failures so callers can catch one
    specific type instead of a raw Chroma/model exception."""


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def get_embedder() -> SentenceTransformer:
    """Loads the embedding model once and reuses it. Imported lazily since
    sentence_transformers pulls in torch, which isn't needed for the OpenAI
    provider."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
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
    """Turns texts into embedding vectors in batches, using whichever
    provider is configured."""
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


def warm_up() -> None:
    """Eagerly opens the vector store and makes a throwaway embedding call,
    so the network connection is already warm before the first question --
    constructing the client alone doesn't touch the slow part (the TLS
    handshake on the first real request)."""
    embed_texts(["warm-up"])
    get_collection()


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
_chroma_client: Optional["chromadb.PersistentClient"] = None


def get_client() -> "chromadb.PersistentClient":
    """Opens the persistent Chroma client once and reuses it."""
    global _chroma_client
    if _chroma_client is None:
        try:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=str(INDEX_DIR))
        except Exception as e:
            raise RetrieverError(f"Failed to open the vector index at {INDEX_DIR}: {e}") from e
    return _chroma_client


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
# Build / rebuild the index
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
    """Drops exact-duplicate chunk text, keeping the first occurrence.
    Returns the deduped chunks along with a content hash per chunk, used
    as stable Chroma IDs."""
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


def _upsert_chunks(collection, chunks: list[dict[str, Any]]) -> int:
    """Validates, dedupes, embeds, and upserts chunks into an existing
    collection. Shared by build_index() and add_document_to_index()."""
    valid_chunks = _validate_chunks(chunks)
    if not valid_chunks:
        return 0

    deduped_chunks, ids = _dedupe_chunks(valid_chunks)
    if not deduped_chunks:
        return 0

    texts = [c[CHUNK_TEXT_KEY] for c in deduped_chunks]
    # Chroma rejects None as a metadata value, so drop any keys set to None.
    metadatas = [
        {k: v for k, v in c.items() if k != CHUNK_TEXT_KEY and v is not None}
        for c in deduped_chunks
    ]

    embeddings = embed_texts(texts)  # already batched internally

    try:
        for start in range(0, len(deduped_chunks), EMBED_BATCH_SIZE):
            end = start + EMBED_BATCH_SIZE
            collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )
    except Exception as e:
        raise RetrieverError(f"Failed to add chunks to the vector store: {e}") from e

    return len(deduped_chunks)


def build_index(chunks: list[dict[str, Any]]) -> int:
    """Embeds and stores chunks in the vector database, replacing whatever
    was indexed before. Invalid or duplicate chunks are skipped with a
    warning rather than raising. Returns the number of chunks actually
    indexed (may be less than len(chunks))."""
    if not chunks:
        return 0

    client = get_client()

    # Drop and recreate the collection so old chunks don't linger.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # didn't exist yet

    try:
        collection = client.get_or_create_collection(COLLECTION_NAME, metadata=COLLECTION_METADATA)
    except Exception as e:
        raise RetrieverError(f"Failed to create collection '{COLLECTION_NAME}': {e}") from e

    return _upsert_chunks(collection, chunks)


def add_document_to_index(path: str | Path) -> int:
    """Loads, chunks, and embeds a single document, then upserts it into
    the existing index -- other documents' chunks are left untouched."""
    from app.chunker import assign_product_name_avoiding, chunk_document
    from app.document_loader import load_document

    document = load_document(Path(path))
    collection = get_collection()

    existing_names = {
        m["product_name"]
        for m in collection.get(include=["metadatas"])["metadatas"]
        if m.get("product_name")
    }
    product_name = assign_product_name_avoiding(document, existing_names)

    chunks = [c.to_dict() for c in chunk_document(document, product_name)]
    return _upsert_chunks(collection, chunks)


def remove_document_from_index(document_name: str) -> None:
    """Removes every chunk belonging to one document from the index,
    without touching any other document's chunks."""
    try:
        get_collection().delete(where={"document_name": document_name})
    except Exception as e:
        raise RetrieverError(f"Failed to remove '{document_name}' from the vector store: {e}") from e


# ---------------------------------------------------------------------------
# "List all products" queries
# ---------------------------------------------------------------------------
# Top-k search answers "which chunks best match this text", not "enumerate
# every product" -- with more products than top_k, one always gets dropped
# arbitrarily. Detected by keyword pattern instead.
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
    global top-k, so every indexed product is represented at least once."""
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
# Product detection -- scopes retrieval to the specific product(s) a query
# names, so a weakly-related chunk from a different product can't outrank
# and bleed into a single-product question's context.
# ---------------------------------------------------------------------------
def _distinctive_product_tokens(product_names: list[str]) -> dict[str, set[str]]:
    """Tokens per product name, excluding common English words and tokens
    shared by most product names (e.g. 'leak', 'detector', 'series') --
    derived from whichever product names are currently indexed, no
    hardcoded product identifiers."""
    all_tokens = {
        name: {tok for tok in re.findall(r"[a-zA-Z0-9]+", name.lower()) if tok not in _STOPWORDS}
        for name in product_names
    }
    doc_frequency = Counter(tok for tokens in all_tokens.values() for tok in tokens)
    max_shared = max(2, len(product_names) // 3)
    return {
        name: {tok for tok in tokens if len(tok) >= 3 and doc_frequency[tok] <= max_shared}
        for name, tokens in all_tokens.items()
    }


def _is_single_product_document(document_name: str) -> bool:
    """Competitor-comparison and other reference documents describe many
    products at once under one umbrella heading (e.g. "Hydrogen Sensor
    Performance Comparison") -- they're valid evidence in ordinary top-k
    retrieval, but that heading must not be eligible for single/multi-
    product name detection below: a query merely containing a generic word
    from it (e.g. "sensor") would otherwise get hard-scoped to only this
    document, silently excluding the actual product catalogue it discusses.
    Proven necessary by testing: "Is the FIXaHY sensor PESO approved..."
    matched "sensor" against this document's own title and hard-scoped
    retrieval to it, hiding the catalogue chunk that actually answers the
    question. Detected by filename pattern since these documents don't
    represent one purchasable product the way a catalogue does."""
    name = document_name.lower()
    return not any(kw in name for kw in ("comparison", "competitor"))


def _detect_mentioned_products(query: str, product_names: list[str]) -> list[str]:
    """Which of the currently indexed products, if any, the query actually
    names -- 0 for a generic question, 1 to scope to that product, 2+ for a
    comparison question spanning exactly those products."""
    query_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    distinctive = _distinctive_product_tokens(product_names)
    return [name for name, tokens in distinctive.items() if tokens & query_words]


def _retrieve_balanced_across_products(
    query_embedding: list[list[float]], collection, product_names: list[str], top_k: int
) -> list[dict[str, Any]]:
    """Retrieves a fair share of chunks per named product instead of one
    shared top-k, so a comparison question isn't silently dominated by
    whichever product's chunks happen to score higher overall -- a single
    $in-filtered query was tried first and confirmed (via real testing,
    not assumption) to let one product crowd out the other entirely. A
    higher floor than plain top_k division matters here specifically: a
    short, differently-phrased spec row (e.g. an "Area of Deployment"
    line) can lose out to a cluster of similar-sounding feature bullets
    at a narrow per-product count, even though it's the more decisive fact."""
    per_product = max(COMPARISON_CHUNKS_PER_PRODUCT, top_k // len(product_names))
    matches: list[dict[str, Any]] = []
    for name in product_names:
        result = collection.query(
            query_embeddings=query_embedding,
            n_results=per_product,
            where={"product_name": name},
        )
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            matches.append({"text": text, "metadata": metadata, "similarity": 1 - distance})
    return matches


# ---------------------------------------------------------------------------
# Keyword boost -- catches chunks that literally contain a query term but
# ranked outside the vector top_k, since a small wording change can shift a
# chunk's relative rank even when its own similarity barely moves.
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "for", "in", "on", "to", "with", "and", "or", "but", "not",
    "this", "that", "these", "those", "it", "its", "we", "our", "you", "your",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "do", "does", "did", "can", "could", "will", "would", "should", "may", "might",
    "i", "me", "my", "us", "if", "so", "as", "at", "by", "from",
}


def _extract_keywords(query: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]


def _keyword_boost_matches(
    query: str,
    query_embedding: list[list[float]],
    collection,
    existing_texts: set[str],
    where: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """Finds chunks that literally contain a query keyword but fell outside
    the vector top_k, and ranks them by similarity among themselves.
    Matching is done in Python (case-insensitive) since Chroma's own
    $contains filter is case-sensitive. Respects the same product scope as
    the caller's main search, if any -- otherwise a keyword match could
    reintroduce the exact cross-product bleed the scope was meant to stop."""
    keywords = _extract_keywords(query)
    if not keywords:
        return []

    all_chunks = collection.get(include=["documents"], where=where)
    candidate_ids = [
        chunk_id
        for chunk_id, text in zip(all_chunks["ids"], all_chunks["documents"])
        if text not in existing_texts and any(kw in text.lower() for kw in keywords)
    ]
    if not candidate_ids:
        return []

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            ids=candidate_ids,
            n_results=min(KEYWORD_BOOST_MAX_EXTRA, len(candidate_ids)),
        )
    except Exception:
        return []  # keyword boost is a nice-to-have, not worth failing retrieval over

    return [
        {"text": text, "metadata": metadata, "similarity": 1 - distance}
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = 8) -> dict[str, Any]:
    """Retrieves the top_k most relevant chunks for a question, plus any
    keyword-matched chunks the vector search missed (see
    _keyword_boost_matches), along with a confidence level ("high", "low",
    or "none") based on similarity."""
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
        all_metadata = collection.get(include=["metadatas"])["metadatas"]
        product_names = sorted({
            m["product_name"]
            for m in all_metadata
            if m.get("product_name") and _is_single_product_document(m.get("document_name", ""))
        })
        mentioned = _detect_mentioned_products(query, product_names)

        if len(mentioned) == 1:
            boost_where = {"product_name": mentioned[0]}
        elif len(mentioned) >= 2:
            boost_where = {"product_name": {"$in": mentioned}}
        else:
            boost_where = None

        try:
            if len(mentioned) >= 2:
                matches = _retrieve_balanced_across_products(query_embedding, collection, mentioned, top_k)
            else:
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(top_k, collection_count),
                    where=boost_where,
                )
                matches = [
                    # Chroma returns cosine distance here (0 = identical);
                    # similarity is the complement, higher is better.
                    {"text": text, "metadata": metadata, "similarity": 1 - distance}
                    for text, metadata, distance in zip(
                        results["documents"][0], results["metadatas"][0], results["distances"][0]
                    )
                ]
        except Exception as e:
            raise RetrieverError(f"Vector search failed: {e}") from e

        existing_texts = {m["text"] for m in matches}
        boosted = _keyword_boost_matches(query, query_embedding, collection, existing_texts, where=boost_where)
        if boosted:
            matches.extend(boosted)
            matches.sort(key=lambda m: m["similarity"], reverse=True)

    # max() not matches[0]: the list-all-products branch orders by document
    # name for readability, not similarity.
    best_similarity = max((m["similarity"] for m in matches), default=0.0)

    if not matches or best_similarity < NO_MATCH_THRESHOLD:
        return {"matches": [], "confidence": "none"}
    elif best_similarity < LOW_CONFIDENCE_THRESHOLD:
        confidence = "low"
    else:
        confidence = "high"

    return {"matches": matches, "confidence": confidence}


# File types the loader can handle -- keep in sync with the loader's own list.
SUPPORTED_DOC_EXTENSIONS = {".docx", ".pptx", ".pdf", ".csv", ".xlsx", ".md", ".markdown", ".txt"}


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
# CLI: rebuild the index, or debug-print retrieve() results for a question.
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
