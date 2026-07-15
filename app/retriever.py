"""
app/retriever.py

Implements the "Embedding / search index" and "Retriever" components from
the spec (System Architecture 4.1) and the rules in Implementation Details
12.1 (Document Ingestion) and 12.2 (Retrieval).

Uses:
- sentence-transformers for embeddings, run locally, no API key needed
  (Already in requirements.txt), OR the OpenAI embeddings API
  (text-embedding-3-small per .env.example) -- see EMBEDDING_PROVIDER
  below and call_openai_embeddings().
- chromadb as the vector store, persisted to disk so the index survives
  between app restarts (also already in requirements.txt).

Expected chunk format coming from chunker.py: a dict per chunk with at least
a "text" key, plus whatever metadata your chunker attaches (source file,
page/slide number, section title, etc.). Adjust CHUNK_TEXT_KEY below if your
chunker uses a different key name for the chunk's text.
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Optional

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Loads variables from a .env file (in the current working directory, or a
# parent of it) into os.environ. Must run before any os.environ.get() calls
# below, or every config value that's only set in .env -- EMBEDDING_PROVIDER,
# OPENAI_API_KEY, VECTOR_DB_PATH, etc. -- silently falls back to its default
# instead of the .env value, with no error to indicate why. Safe to call even
# if no .env file exists (no-op), and never overwrites a variable that's
# already set in the real environment (e.g. by the OS or a deploy platform).
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNK_TEXT_KEY = "text"
# Per .env: VECTOR_DB_PATH=./vector_store/chroma. Falls back to the path
# below if VECTOR_DB_PATH isn't set, same pattern as EMBEDDING_MODEL.
INDEX_DIR = Path(os.environ.get("VECTOR_DB_PATH", "data/chroma_index"))
COLLECTION_NAME = "approved_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # local sentence-transformers model name; small and fast

# "local" (default, no API key needed) or "openai" (uses
# call_openai_embeddings() below). Overridable via env var so switching
# providers is a config change, not a code change.
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")

# OpenAI embedding model, used only when EMBEDDING_PROVIDER="openai".
# Per .env.example: EMBEDDING_MODEL=text-embedding-3-small.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# How many texts get embedded / added to Chroma per batch, rather than all
# at once. Keeps memory bounded regardless of how many chunks you have --
# 20,000 chunks embedded in one call means 20,000 texts and their vectors
# all resident in memory simultaneously; batching processes and discards
# them in smaller waves instead. TODO: move to a config file alongside the
# thresholds below once you have more than a couple of tunables like this.
EMBED_BATCH_SIZE = 64

# Per spec 12.2: "Do not answer if no relevant source is found." Below this
# similarity score, treat retrieval as a miss. Overridable via env var so
# you can tune without a code change; TODO: move into a proper config file
# once there are enough of these to warrant one -- not urgent at this size.
NO_MATCH_THRESHOLD = float(os.environ.get("RETRIEVER_NO_MATCH_THRESHOLD", 0.35))

# Per spec 12.2: "If retrieved chunks have low similarity or weak relevance,
# answer with low confidence." Between NO_MATCH_THRESHOLD and this, answer
# but flag confidence as low. Also overridable via env var.
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("RETRIEVER_LOW_CONFIDENCE_THRESHOLD", 0.55))

_embedder: Optional[SentenceTransformer] = None


class RetrieverError(RuntimeError):
    """Raised when an embedding or vector-store operation fails, wrapping
    the underlying exception with context about what was being attempted.
    Callers (e.g. the Streamlit app) can catch this one type to show a
    friendly message instead of letting a raw Chroma/model stack trace
    crash the whole page."""


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def get_embedder() -> SentenceTransformer:
    """Loads the embedding model once and reuses it (loading it per-call is slow)."""
    global _embedder
    if _embedder is None:
        try:
            _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as e:
            raise RetrieverError(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}") from e
    return _embedder


_openai_client = None


def _get_openai_client():
    """Loads the OpenAI client once and reuses it (same singleton pattern
    as get_embedder() above for the local model). Reads OPENAI_API_KEY
    from the environment only -- never hardcode a key here."""
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
    """Calls the OpenAI embeddings API for one batch of texts, using the
    company-approved OPENAI_API_KEY and EMBEDDING_MODEL from .env
    (default: text-embedding-3-small).

    embed_texts() already batches calls via EMBED_BATCH_SIZE, so `texts`
    here is one batch, not the full document set.
    """
    client = _get_openai_client()
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Embeds one batch of texts using whichever provider EMBEDDING_PROVIDER
    selects. This is the single place embed_texts() calls out to per batch,
    so the batching/error-wrapping logic in embed_texts() doesn't need to
    change no matter which provider is active."""
    if EMBEDDING_PROVIDER == "openai":
        return call_openai_embeddings(batch)
    return get_embedder().encode(batch, show_progress_bar=False).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Turns a list of strings into a list of embedding vectors, in batches
    of EMBED_BATCH_SIZE rather than all at once -- keeps memory bounded
    for a large index build instead of holding every text and its vector
    in memory simultaneously.

    Provider is controlled by EMBEDDING_PROVIDER ("local" by default, using
    sentence-transformers; "openai" to use call_openai_embeddings() with
    the EMBEDDING_MODEL from .env). build_index() and retrieve() don't need
    to know or care which provider is active -- this function is the only
    place that decision is made.
    """
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
        return client.get_or_create_collection(COLLECTION_NAME)
    except Exception as e:
        raise RetrieverError(f"Failed to open collection '{COLLECTION_NAME}': {e}") from e


def index_size() -> int:
    """Handy for the sidebar 'Indexed docs' style status, or for tests.
    Returns 0 on any failure rather than raising -- a status display
    shouldn't crash the page just because the index isn't built yet."""
    try:
        return get_collection().count()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Build / rebuild the index (spec 12.1)
# ---------------------------------------------------------------------------
def _validate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filters out chunks missing a usable CHUNK_TEXT_KEY, instead of
    letting a malformed chunk (e.g. from a chunker bug, or a caller
    passing the wrong dict shape) blow up build_index with a bare
    KeyError partway through. Prints a warning for each one skipped, the
    same way document_loader.py surfaces its own coverage warnings."""
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
    """Drops chunks whose text is an exact duplicate of one already seen
    (keeping the first occurrence), so identical chunks -- e.g. from
    re-running the chunker over an unchanged document, or the same
    boilerplate paragraph appearing in two source files -- don't end up
    indexed twice and returned as duplicate search results. Content-based,
    not id-based: two chunks with identical text but different metadata
    (different source file) are still considered duplicates here, since
    what matters for "don't show the same passage twice" is the text
    itself. Returns (deduped_chunks, content_hash_per_chunk) -- the hashes
    double as stable, deterministic Chroma IDs in build_index."""
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
    """
    Embeds and stores chunks in the vector database, replacing whatever was
    indexed before. Call this whenever documents change (spec 12.1: "Rebuild
    the index when documents change").

    chunks: list of dicts, e.g.
        {"text": "...", "source": "product_positioning.md", "page": 2, "section": "Target applications"}
    Any keys other than "text" are stored as metadata and come back attached
    to each result from retrieve(), which is what your UI uses to show
    "sources" (spec 12.2: "Show the sources used in the answer.").

    Chunks missing a usable "text" value are skipped with a warning rather
    than raising. Exact-duplicate chunk text is also skipped (see
    _dedupe_chunks) so identical passages don't get indexed twice.

    Raises RetrieverError if the vector store or embedding step fails --
    lets a caller catch one specific exception type instead of a raw
    Chroma/model exception.

    Returns the number of chunks actually indexed (after validation and
    deduping -- may be less than len(chunks)).
    """
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
        collection = client.get_or_create_collection(COLLECTION_NAME)
    except Exception as e:
        raise RetrieverError(f"Failed to create collection '{COLLECTION_NAME}': {e}") from e

    texts = [c[CHUNK_TEXT_KEY] for c in deduped_chunks]
    # Chroma rejects None as a metadata value (raises on collection.add), so
    # drop any key whose value is None rather than sending it through --
    # e.g. a chunk with no slide_number (a docx/pdf/csv chunk, not pptx)
    # naturally has page_number/slide_number as None from Chunk.to_dict().
    # A missing key behaves the same as a None one for any caller using
    # metadata.get(...), which is the only safe way to read it anyway.
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
# Retrieval (spec 12.2)
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Retrieves the most relevant chunks for a question.

    Spec 12.2 rules implemented here:
    - "For each question, retrieve top 3 to 5 relevant chunks." -> top_k
    - "If retrieved chunks have low similarity or weak relevance, answer
      with low confidence." -> confidence == "low"
    - "Do not answer if no relevant source is found." -> confidence == "none",
      matches will be empty; your answer generator should check for this
      and return a "no approved source found" response instead of guessing.

    Raises RetrieverError if the vector store or embedding step fails.

    Returns:
        {
            "matches": [
                {"text": "...", "metadata": {"source": ..., "page": ..., ...}, "similarity": 0.0-1.0},
                ...
            ],
            "confidence": "high" | "low" | "none",
        }
    """
    collection = get_collection()

    try:
        collection_count = collection.count()
    except Exception as e:
        raise RetrieverError(f"Failed to read the vector store: {e}") from e

    if collection_count == 0:
        return {"matches": [], "confidence": "none"}

    query_embedding = embed_texts([query])

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
        # Chroma returns cosine distance by default (0 = identical); convert
        # to a similarity score where higher is better, for readability.
        similarity = 1 - distance
        matches.append({"text": text, "metadata": metadata, "similarity": similarity})

    best_similarity = matches[0]["similarity"] if matches else 0.0

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
# Kept in sync by hand with document_loader.py's _EXTRACTORS keys. Not
# imported directly from there since it's a private module constant --
# duplicating a short list here is simpler than exposing it as public API
# just for this.
SUPPORTED_DOC_EXTENSIONS = {".docx", ".pptx", ".pdf", ".csv", ".md", ".markdown", ".txt"}


def load_and_chunk_approved_docs(docs_dir: str | Path = "data/approved_docs") -> list[dict[str, Any]]:
    """Loads every supported file in docs_dir via document_loader.load_document,
    chunks all of them via chunker.chunk_documents, and returns plain dicts
    ready for build_index() (via Chunk.to_dict()).

    document_loader/chunker are imported here rather than at module level,
    so retriever.py doesn't hard-require them just to use embed_texts() /
    build_index() / retrieve() directly -- e.g. from a test that constructs
    chunk dicts itself without touching real files.

    A file that fails to load (corrupt, unsupported despite the extension,
    etc.) is skipped with a warning rather than aborting the whole reindex --
    matches document_loader's own "surface problems, don't crash on them"
    approach.
    """
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
# CLI: rebuild the index from your loader + chunker
# ---------------------------------------------------------------------------
if __name__ == "__main__":
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