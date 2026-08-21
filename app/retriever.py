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


def all_document_names() -> list[str]:
    """Every distinct document name currently indexed. For a caller that
    needs to exclude some subset of documents from retrieve() (via
    exclude_document_names) but only knows the exclusion *rule*, not the
    concrete file list -- letting it filter this itself rather than
    duplicating Chroma access to work that out independently."""
    try:
        metadatas = get_collection().get(include=["metadatas"])["metadatas"]
    except Exception:
        return []
    return sorted({m["document_name"] for m in metadatas if m.get("document_name")})


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


def add_document_to_index(path: str | Path) -> tuple[int, list[str]]:
    """Loads, chunks, and embeds a single document, then upserts it into
    the existing index -- other documents' chunks are left untouched.
    Returns (chunk_count, warnings), surfacing load_document()'s own
    coverage/structure checks (dropped cells, implausible header
    detection) to the admin instead of discarding them silently."""
    from app.chunker import assign_product_name_avoiding, chunk_approved_claims, chunk_document
    from app.claim_checker import APPROVED_CLAIMS_DOCUMENT_NAME
    from app.document_loader import load_document

    path = Path(path)
    collection = get_collection()

    if path.name == APPROVED_CLAIMS_DOCUMENT_NAME:
        # One bullet per chunk -- see chunk_approved_claims's own
        # docstring for why chunk_document's general word-count packing
        # is wrong for this file specifically.
        from app.claims_store import load_claims

        _, bullets = load_claims(path)
        chunks = [c.to_dict() for c in chunk_approved_claims(path.name, bullets)]
        count = _upsert_chunks(collection, chunks)
        return count, []

    document = load_document(path)

    from app.product_name_overrides import load_overrides as load_product_name_overrides

    override = load_product_name_overrides().get(path.name)
    if override:
        product_name = override
    else:
        existing_names = {
            m["product_name"]
            for m in collection.get(include=["metadatas"])["metadatas"]
            if m.get("product_name")
        }
        product_name = assign_product_name_avoiding(document, existing_names)

    chunks = [c.to_dict() for c in chunk_document(document, product_name)]
    count = _upsert_chunks(collection, chunks)
    return count, document["warnings"]


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
_PRODUCT_OR_CATEGORY_NOUN = r"(?:products?|sensors?|detectors?|devices?|analyzers?|units?)"
_LIST_ALL_PRODUCTS_PATTERNS = [
    re.compile(r"\ball\b.{0,20}\b" + _PRODUCT_OR_CATEGORY_NOUN + r"\b", re.IGNORECASE),
    re.compile(
        r"\bwhat\b.{0,10}\b" + _PRODUCT_OR_CATEGORY_NOUN + r"\b.{0,15}\b(available|offer|have|sell|make)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b" + _PRODUCT_OR_CATEGORY_NOUN + r"\s+(lineup|line-up|catalog|catalogue|range|portfolio)\b", re.IGNORECASE),
    re.compile(r"\blist\b.{0,20}\b" + _PRODUCT_OR_CATEGORY_NOUN + r"\b", re.IGNORECASE),
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
def _product_name_tokens(product_names: list[str]) -> dict[str, set[str]]:
    """Tokens per product name, excluding common English words -- derived
    from whichever product names are currently indexed, no hardcoded
    product identifiers."""
    return {
        name: {tok for tok in re.findall(r"[a-zA-Z0-9]+", name.lower()) if tok not in _STOPWORDS}
        for name in product_names
    }


def _distinctive_product_tokens(product_names: list[str]) -> dict[str, set[str]]:
    """Tokens per product name that uniquely (or near-uniquely) identify
    it -- excludes tokens shared by most product names (e.g. 'leak',
    'detector', 'series')."""
    all_tokens = _product_name_tokens(product_names)
    doc_frequency = Counter(tok for tokens in all_tokens.values() for tok in tokens)
    max_shared = max(2, len(product_names) // 3)
    return {
        name: {tok for tok in tokens if len(tok) >= 3 and doc_frequency[tok] <= max_shared}
        for name, tokens in all_tokens.items()
    }


def _shared_family_tokens(product_names: list[str]) -> dict[str, set[str]]:
    """Complement of _distinctive_product_tokens(): tokens shared across
    multiple products (e.g. "FIXaHY") -- still a real brand-family
    reference, just not specific enough to narrow to one product. Maps
    each such token to the product names that share it."""
    all_tokens = _product_name_tokens(product_names)
    doc_frequency = Counter(tok for tokens in all_tokens.values() for tok in tokens)
    max_shared = max(2, len(product_names) // 3)
    shared: dict[str, set[str]] = {}
    for name, tokens in all_tokens.items():
        for tok in tokens:
            if len(tok) >= 3 and doc_frequency[tok] > max_shared:
                shared.setdefault(tok, set()).add(name)
    return shared


def _is_single_product_document(document_name: str) -> bool:
    """True only for real product catalogues. Reference documents (a
    competitor comparison, a sales log) also carry one umbrella heading,
    but a generic word from that heading (e.g. "sensor") must not
    hard-scope a query to just that document instead of the real
    catalogues. Driven by the Admin-assigned document type, not a
    filename heuristic, which would be too easy to fool."""
    from app.document_types import is_product_catalogue

    return is_product_catalogue(document_name)


def main_assistant_excluded_document_names() -> set[str]:
    """Documents excluded from the main Assistant (see
    app.document_types.MAIN_ASSISTANT_EXCLUDED_TYPES) -- internal
    sales-strategy content is MNST's own subjective judgment, not
    verified fact, so it may feed Sales Aid (guardrail-checked) or
    Discovery (internal-only) but never the Assistant's unchecked
    rep-facing answers. rag_pipeline.py applies this set; Sales Aid and
    Discovery call retrieve() directly and are unaffected."""
    from app.document_types import is_excluded_from_main_assistant

    return {name for name in all_document_names() if is_excluded_from_main_assistant(name)}


def _detect_mentioned_products(query: str, product_names: list[str]) -> list[str]:
    """Which indexed products, if any, the query names -- 0 for generic,
    1 to scope to that product, 2+ for a comparison. Naming one distinctive
    product plus a shared family prefix (e.g. "FIXaHY") pulls in every
    product sharing that prefix too. A family prefix alone, with nothing
    distinctive alongside it, stays unscoped -- see
    is_ambiguous_product_reference for that case."""
    query_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    distinctive = _distinctive_product_tokens(product_names)
    mentioned = {name for name, tokens in distinctive.items() if tokens & query_words}

    if mentioned:
        shared = _shared_family_tokens(product_names)
        for tok, names in shared.items():
            if tok in query_words:
                mentioned.update(names)

    return sorted(mentioned)


def _product_names_for_scoping(collection) -> list[str]:
    """Currently indexed single-product-document names -- the shared pool
    every product-detection entry point (retrieve()'s scoping,
    is_ambiguous_product_reference()) draws from."""
    all_metadata = collection.get(include=["metadatas"])["metadatas"]
    return sorted({
        m["product_name"]
        for m in all_metadata
        if m.get("product_name") and _is_single_product_document(m.get("document_name", ""))
    })


def _mentioned_products_for_query(query: str, collection) -> list[str]:
    """Which currently-indexed products the query names -- see
    _detect_mentioned_products()."""
    return _detect_mentioned_products(query, _product_names_for_scoping(collection))


# Generic English words that refer to "a product" without naming one --
# closed set, not catalogue-dependent. Allows up to two words between
# "the" and the noun ("the hydrogen sensor"); whether that phrase actually
# names a product is decided separately by is_ambiguous_product_reference().
_AMBIGUOUS_REFERENCE_RE = re.compile(
    r"\b(this|these|it)\b|\bthe (?:\w+\s+){0,2}(product|device|unit|sensor|detector|instrument|system)\b",
    re.IGNORECASE,
)

# Same shape as _AMBIGUOUS_REFERENCE_RE ("the [word] product") but a
# different intent -- ranking across the catalog, not confusion about
# which product is meant -- so it must not trigger a clarification ask.
_SUPERLATIVE_RE = re.compile(
    r"\b(most|least)\s+\w+|\b\w{4,}est\b|\b(better|best|worse|worst)\b",
    re.IGNORECASE,
)


def _brand_root(name: str) -> str:
    """First token of a product name, used as a coarse proxy for 'brand
    family' -- e.g. FIXaHY Analyzer Series and FIXaHY H2 LD XX share the
    root 'fixahy', while VISION H2 LD XX doesn't, even though both
    mention 'H2 LD'."""
    tokens = re.findall(r"[a-zA-Z0-9]+", name.lower())
    return tokens[0] if tokens else name.lower()


def _product_types_from_registry() -> dict[str, str]:
    """Product name -> lowercased product_type ("leak detector"/
    "analyzer"), read from the same auto-extracted Product Index
    dispatcher.py and registry_builder.py already maintain -- not a
    second, hand-maintained mapping that could drift out of sync with
    the real product names. registry_builder's "Leak Detector"/"Analyzer"
    vocab is lowercased here to match this module's own comparison
    convention. Falls back to an empty mapping if the registry isn't
    built yet, same as every other registry-dependent check in this
    codebase."""
    try:
        from app.product_index import load_product_index
        products, _ = load_product_index()
    except (FileNotFoundError, OSError):
        return {}
    return {p.product_name: p.product_type.lower() for p in products if p.product_type != "Unknown"}


# How a category can be recognized in a query, mapped to the canonical
# value _product_types_from_registry() returns -- kept separate so the
# recognizer can accept a few natural variants (bare "detector", plurals)
# without those variants needing to be a product's literal stored type.
_PRODUCT_TYPE_TERMS: dict[str, str] = {
    "leak detector": "leak detector",
    "leak detectors": "leak detector",
    "leak detection": "leak detector",
    "detector": "leak detector",
    "detectors": "leak detector",
    "analyzer": "analyzer",
    "analyzers": "analyzer",
}


def _product_type_ambiguous(query: str, product_names: list[str]) -> bool:
    """True if the query names a product CATEGORY (e.g. "leak detector")
    shared by 2+ indexed products, without naming a specific product/brand.
    Catches e.g. "the leak detector" resolving to AURIGA just because its
    name spells that phrase out in full, while FIXaHY/PORTaHY/VISION
    abbreviate it as "H2 LD" -- a naming accident, not a real distinction."""
    query_lower = query.lower()
    query_words = set(re.findall(r"[a-zA-Z0-9]+", query_lower))
    types_present = {canonical for term, canonical in _PRODUCT_TYPE_TERMS.items() if term in query_lower}
    if not types_present:
        return False

    product_types = _product_types_from_registry()
    all_tokens = _product_name_tokens(product_names)
    for product_type in types_present:
        matching = [name for name in product_names if product_types.get(name) == product_type]
        if len(matching) < 2:
            continue

        # One of the matching products named completely -- not ambiguous.
        if any(all_tokens[name] and all_tokens[name] <= query_words for name in matching):
            continue

        # A brand was named that narrows to exactly one matching product.
        brands = {_brand_root(name) for name in matching}
        named_brands = brands & query_words
        if len(named_brands) == 1:
            within = [n for n in matching if _brand_root(n) == next(iter(named_brands))]
            if len(within) == 1:
                continue

        return True
    return False


def _cross_brand_ambiguous(query: str, product_names: list[str]) -> bool:
    """True if the query's words span 2+ DIFFERENT brand families without
    naming any single product completely or any brand directly. Catches
    e.g. "size of H2 LD" -- "H2"/"LD" are too short/generic to register as
    distinctive tokens, but the phrase still spans materially different
    products (FIXaHY H2 LD XX vs. VISION H2 LD XX). Unlike "the FIXaHY
    sensor" (one brand) or "FIXaHY vs PORTaHY" (both named on purpose)."""
    query_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    all_tokens = _product_name_tokens(product_names)

    # A product counts as fully named if every one of its tokens appears
    # in the query -- short/generic tokens can still pin down one exact
    # product when combined (e.g. "FIXaHY H2 LD XX" needs all four
    # together, no single one of which is distinctive alone).
    if any(tokens and tokens <= query_words for tokens in all_tokens.values()):
        return False

    matched_products = {name for name, tokens in all_tokens.items() if tokens & query_words}
    matched_brands = {_brand_root(name) for name in matched_products}
    if len(matched_brands) < 2:
        return False

    brand_roots = {_brand_root(name) for name in product_names}
    directly_named_brands = brand_roots & query_words
    if len(directly_named_brands) >= 2:
        return False  # explicit multi-brand comparison, e.g. "FIXaHY vs PORTaHY"

    if len(directly_named_brands) == 1:
        # A brand was named directly, so the cross-brand overlap above is
        # incidental. Check if the query narrows to one clear variant
        # within that brand (e.g. "FIXaHY H2 LD" matches FIXaHY H2 LD XX
        # on 3/4 tokens, vs. 1 for other FIXaHY products).
        named_brand = next(iter(directly_named_brands))
        overlaps = {
            name: len(all_tokens[name] & query_words)
            for name in product_names
            if _brand_root(name) == named_brand
        }
        max_overlap = max(overlaps.values())
        best = [name for name, count in overlaps.items() if count == max_overlap]
        if len(best) == 1 and max_overlap > 1:
            return False  # a clear single best match within the named brand

    return True


def is_ambiguous_product_reference(query: str) -> bool:
    """True if the query refers to a product generically ("this", "the
    sensor") without naming a real one, or spans multiple brand families
    via generic shared terms (_cross_brand_ambiguous), or uses a category
    term shared by multiple products (_product_type_ambiguous). Retrieval
    can't resolve any of these on its own -- a topical match isn't
    confirmation of which product is meant, so callers should ask for
    clarification. Exceptions: a shared family prefix ("the FIXaHY
    sensor") still names something real; a "list all X" query
    (_is_list_all_products_query) deliberately wants every match, so it's
    never ambiguous even when it spans multiple brands."""
    if _is_list_all_products_query(query):
        return False
    if _SUPERLATIVE_RE.search(query):
        return False
    collection = get_collection()
    if collection.count() == 0:
        return False
    product_names = _product_names_for_scoping(collection)

    if _cross_brand_ambiguous(query, product_names):
        return True
    if _product_type_ambiguous(query, product_names):
        return True

    if not _AMBIGUOUS_REFERENCE_RE.search(query):
        return False
    if _detect_mentioned_products(query, product_names):
        return False
    query_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    shared = _shared_family_tokens(product_names)
    return not any(tok in query_words for tok in shared)


_SELF_REFERENTIAL_RE = re.compile(
    r"\b(our|ours|we|we're|we are|us|the company)\b",
    re.IGNORECASE,
)


def is_self_referential_without_own_products(
    query: str, matches: list[dict[str, Any]]
) -> bool:
    """True if the query asks about "our"/"we"/"us"/"the company" (this
    company's own products, positioning, capabilities, offerings...) but
    every retrieved match comes from a reference document (competitor
    comparison, historical sales log, etc. -- see
    _is_single_product_document) rather than an actual product catalogue.
    A self-referential question retrieved entirely from a competitor
    document risks the model describing the competitor's own products as
    this company's -- an attribution error a prompt instruction alone
    can't reliably prevent, so it needs a deterministic check on what was
    actually retrieved."""
    if not _SELF_REFERENTIAL_RE.search(query):
        return False
    if not matches:
        return False
    return not any(
        _is_single_product_document(m["metadata"].get("document_name", ""))
        for m in matches
    )


def _interleave_balanced_matches(
    matches: list[dict[str, Any]], product_names: list[str]
) -> list[dict[str, Any]]:
    """Round-robins matches across product_names (best-of-product-1,
    best-of-product-2, ..., second-best-of-product-1, ...) instead of one
    block per product or a plain similarity sort -- either would bury a
    lower-scoring product's chunks deep enough that the model only
    discusses whichever products it saw first."""
    by_product: dict[str, list[dict[str, Any]]] = {}
    for m in matches:
        by_product.setdefault(m["metadata"].get("product_name"), []).append(m)
    for group in by_product.values():
        group.sort(key=lambda m: m["similarity"], reverse=True)

    ordered_keys = list(product_names) + [k for k in by_product if k not in product_names]
    interleaved: list[dict[str, Any]] = []
    index = 0
    while True:
        added_any = False
        for name in ordered_keys:
            group = by_product.get(name, [])
            if index < len(group):
                interleaved.append(group[index])
                added_any = True
        if not added_any:
            break
        index += 1
    return interleaved


def _retrieve_balanced_across_products(
    query_embedding: list[list[float]], collection, product_names: list[str], top_k: int
) -> list[dict[str, Any]]:
    """Retrieves a fair share of chunks per named product instead of one
    shared top-k, so a comparison isn't dominated by whichever product's
    chunks score higher overall. Uses a higher floor than plain top_k
    division since a short, differently-phrased spec row can otherwise
    lose out to a cluster of similar-sounding feature bullets."""
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
    the vector top_k. Matches in Python (case-insensitive) since Chroma's
    own $contains filter is case-sensitive. Respects the caller's product
    scope, if any, so it can't reintroduce cross-product bleed."""
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


def _merge_where(where: Optional[dict], extra: Optional[dict]) -> Optional[dict]:
    """Combines two Chroma `where` filters with AND. Either may be None."""
    if where is None:
        return extra
    if extra is None:
        return where
    return {"$and": [where, extra]}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    top_k: int = 8,
    exclude_document_names: Optional[set[str]] = None,
    scope_to_products: bool = True,
    mentioned_products: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Retrieves the top_k most relevant chunks for a question, plus
    keyword-matched chunks the vector search missed, with a confidence
    level ("high"/"low"/"none") based on similarity.

    exclude_document_names filters out of the vector search itself, not
    just the results -- otherwise a dominant excluded document could
    occupy the whole top_k regardless of how wide it's fetched.

    scope_to_products=False skips single-product narrowing, for callers
    needing breadth across multiple MNST products plus non-product
    documents in one call (e.g. sales_aid_generator.py's competitor
    comparisons).

    mentioned_products, when given, overrides this function's own product
    detection -- for a caller (the dispatcher) with better resolution
    than exact-token matching (e.g. typo tolerance: "porthay")."""
    collection = get_collection()

    try:
        collection_count = collection.count()
    except Exception as e:
        raise RetrieverError(f"Failed to read the vector store: {e}") from e

    if collection_count == 0:
        return {"matches": [], "confidence": "none"}

    exclude_where = (
        {"document_name": {"$nin": sorted(exclude_document_names)}} if exclude_document_names else None
    )

    query_embedding = embed_texts([query])

    # Checked only when the caller hasn't already resolved products: a
    # query like "compare auriga and all fixahy products" still contains
    # "all"/"products" and would otherwise be misread as "list everything",
    # discarding the dispatcher's correct product scoping.
    if mentioned_products is None and _is_list_all_products_query(query):
        try:
            matches = _retrieve_one_per_document(query_embedding, collection)
        except Exception as e:
            raise RetrieverError(f"Vector search failed: {e}") from e
    else:
        if mentioned_products is not None:
            mentioned = mentioned_products
        else:
            mentioned = _mentioned_products_for_query(query, collection) if scope_to_products else []

        if len(mentioned) == 1:
            boost_where = {"product_name": mentioned[0]}
        elif len(mentioned) >= 2:
            boost_where = {"product_name": {"$in": mentioned}}
        else:
            boost_where = None
        boost_where = _merge_where(boost_where, exclude_where)

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

        if len(mentioned) >= 2:
            # _retrieve_balanced_across_products' output is contiguous
            # blocks per product; interleave so no product is buried, and
            # never plain-sort (that would undo the fairness just built).
            matches = _interleave_balanced_matches(matches, mentioned)
        elif boosted:
            matches.sort(key=lambda m: m["similarity"], reverse=True)

        if len(mentioned) >= 1:
            # Append an unscoped supplemental pass (product_name filter
            # dropped, exclude_where kept) rather than mixing it into the
            # ranked order. Needed because product_name-scoped queries can
            # never surface a reference document (e.g. the Industry Use
            # Case Guide, or Approved Claims content for "what is
            # PORTaHY's recommended probe length") since those chunks
            # aren't tagged with any product name. No-op when exclude_where
            # already rules reference documents out (see
            # dispatcher._scoped_result).
            try:
                supplement = collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(top_k, collection_count),
                    where=exclude_where,
                )
                existing_texts = {m["text"] for m in matches}
                for text, metadata, distance in zip(
                    supplement["documents"][0], supplement["metadatas"][0], supplement["distances"][0]
                ):
                    if text not in existing_texts:
                        matches.append({"text": text, "metadata": metadata, "similarity": 1 - distance})
                        existing_texts.add(text)
            except Exception:
                pass  # the scoped matches already found are enough to proceed on

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
    """Loads and chunks every supported file in docs_dir, plus the
    Approved Claims reference file (app.claim_checker.
    APPROVED_CLAIMS_DOCUMENT_NAME) -- it lives outside docs_dir and the
    Documents tab, but must still survive a full rebuild, not just the
    Approved Claims tab's own one-off reindex. Returns plain dicts ready
    for build_index(); a file that fails to load is skipped with a
    warning rather than aborting the whole reindex."""
    from app.chunker import chunk_approved_claims, chunk_documents
    from app.claim_checker import APPROVED_CLAIMS_DOCUMENT_NAME
    from app.claims_store import load_claims
    from app.document_loader import load_document

    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        print(f"WARNING: {docs_dir} does not exist -- nothing to index.", file=sys.stderr)

    documents: list[dict[str, Any]] = []
    paths = sorted(docs_dir.iterdir()) if docs_dir.exists() else []

    for path in paths:
        # "~$..." lock files Word/Excel/PowerPoint drop next to an open
        # document -- excluded outright to avoid a wasted load attempt.
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_DOC_EXTENSIONS or path.name.startswith("~$"):
            continue
        try:
            documents.append(load_document(path))
        except Exception as e:
            print(f"WARNING: failed to load {path.name}: {e}", file=sys.stderr)

    chunks = chunk_documents(documents) if documents else []

    # Approved Claims chunked separately, one bullet per chunk -- see
    # chunk_approved_claims's own docstring for why the general word-
    # count packing above is wrong for this file specifically.
    approved_claims_path = Path("data") / APPROVED_CLAIMS_DOCUMENT_NAME
    if approved_claims_path.exists():
        _, bullets = load_claims(approved_claims_path)
        if bullets:
            chunks.extend(chunk_approved_claims(APPROVED_CLAIMS_DOCUMENT_NAME, bullets))

    if not chunks:
        print(f"WARNING: no supported documents found in {docs_dir}.", file=sys.stderr)
        return []

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
