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
    """The complement of _distinctive_product_tokens(): tokens shared
    across multiple product names -- excluded there because they don't
    narrow to one specific product, but they're still a real brand-family
    reference (e.g. "FIXaHY", spanning several variants), not nothing.
    Maps each such token to the product names that share it."""
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
    """Competitor-comparison, historical-sales, and other reference
    documents describe many things at once under one umbrella heading
    (e.g. "Hydrogen Sensor Performance Comparison", or a sales log titled
    "...Customer and Use Case with Value") -- they're valid evidence in
    ordinary top-k retrieval, but that heading must not be eligible for
    single/multi-product name detection below: a query merely containing
    a generic word from it (e.g. "sensor", or an ordinary discovery
    question naturally containing "customer"/"use case") would otherwise
    get hard-scoped to only that document, silently excluding the actual
    product catalogues it should be searching instead. Proven necessary
    by testing twice: "Is the FIXaHY sensor PESO approved..." matched
    "sensor" against the comparison document's own title; a Discovery
    recommendation query matched "customer"/"use case" against the sales
    log's own title, hard-scoping an entire recommendation to a
    spreadsheet of past deals and finding nothing.

    Driven by the explicit Admin-assigned document type (app.document_types),
    not filename pattern-matching -- the old filename-keyword heuristic
    ("eligible unless the name contains comparison/competitor/guide, or
    ends in .xlsx") silently misclassified two different reference
    documents as products this session, purely because their filenames
    didn't happen to contain the magic keyword."""
    from app.document_types import is_product_catalogue

    return is_product_catalogue(document_name)


def main_assistant_excluded_document_names() -> set[str]:
    """Every currently-indexed document whose Admin-assigned type is
    excluded from the main Assistant (see
    app.document_types.MAIN_ASSISTANT_EXCLUDED_TYPES) -- internal sales-
    strategy documents hold MNST's own subjective judgment (self-scored
    competitive positioning, per-vertical objections/stakeholder notes)
    rather than independently verified fact, and are meant to feed sales-
    prep tools where either a claim-checking guardrail sits between the
    content and a customer (Sales Aid) or the output stays internal,
    rep-facing prep (Discovery Questions) -- never the main Assistant,
    where a retrieved chunk goes straight into a rep-visible answer with
    no such check. sales_aid_generator.py and discovery_generator.py both
    call retrieve() directly (not through rag_pipeline.py), so they're
    unaffected by this; rag_pipeline.py adds this set to every retrieve()
    call the main Assistant makes."""
    from app.document_types import is_excluded_from_main_assistant

    return {name for name in all_document_names() if is_excluded_from_main_assistant(name)}


def _detect_mentioned_products(query: str, product_names: list[str]) -> list[str]:
    """Which of the currently indexed products, if any, the query actually
    names -- 0 for a generic question, 1 to scope to that product, 2+ for a
    comparison question spanning exactly those products. A query that
    names one distinctive product AND a shared family prefix (e.g. "FIXaHY",
    spanning several variants -- "difference between FIXaHY and PORTaHY")
    pulls in every product sharing that prefix too, so a family mention
    alongside a specific product isn't silently dropped from retrieval;
    a family prefix with nothing else distinctive alongside it stays
    unscoped, unchanged (see is_ambiguous_product_reference for that case,
    which is handled separately -- broad-but-real is not "name a product
    for scoping purposes")."""
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
    """Single-product-document names (see _is_single_product_document)
    currently indexed -- the pool every product-detection entry point
    (retrieve()'s scoping, is_ambiguous_product_reference()) draws from,
    so there's one place that knows how to go from "the live collection"
    to "which product names exist," no second, divergent copy."""
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


# Closed set of generic English words a question can use to refer back to
# "a product" without naming one -- not product names or domain terms, so
# this doesn't need to change as the catalogue changes. Allows up to two
# words between "the" and the noun ("the hydrogen sensor") since a
# genuinely generic descriptor there doesn't name anything either; whether
# that phrase turns out to actually name a product (e.g. "the FIXaHY
# sensor") is decided separately, by is_ambiguous_product_reference().
_AMBIGUOUS_REFERENCE_RE = re.compile(
    r"\b(this|these|it)\b|\bthe (?:\w+\s+){0,2}(product|device|unit|sensor|detector|instrument|system)\b",
    re.IGNORECASE,
)

# "the lightest product", "the cheapest sensor", "the most accurate
# detector" -- grammatically the same shape _AMBIGUOUS_REFERENCE_RE
# matches ("the [word] product"), but a fundamentally different intent:
# not confusion about which single, already-existing thing is meant, but
# a deliberate request to rank/compare across the whole catalog. Proven
# necessary by testing: "which is the lightest product MNST sells" hit
# the generic "Could you specify which product?" message, which doesn't
# even make sense as a response to a question that already covers every
# product by construction.
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
    second, hand-maintained mapping. Proven necessary by testing: a
    hand-maintained dict here (see git history) had silently drifted out
    of sync with the real, auto-extracted product names -- still keyed
    "AURIGA PORTABLE H2 LEAK DETECTOR" and "PORTaHY SERIES" instead of
    the actual current names "AURIGA" and "PORTaHY H2 LD XX" -- so a
    fully-named AURIGA question ("...a portable or a fixed detector?")
    wasn't recognized as already naming a leak detector, and the
    "detector" reference got wrongly flagged ambiguous between the two
    OTHER products this stale dict still recognized. registry_builder's
    "Leak Detector"/"Analyzer" vocab is lowercased here to match this
    module's own comparison convention. Falls back to an empty mapping if
    the registry isn't built yet, same as every other registry-dependent
    check in this codebase."""
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
    that 2+ currently-indexed products share, without the query naming a
    specific product or brand distinctly enough to narrow to one of them.
    This is the fix for e.g. "the leak detector" resolving to AURIGA
    purely because AURIGA's name happens to spell that phrase out in full
    while FIXaHY/PORTaHY/VISION abbreviate it as "H2 LD" -- a naming
    accident, not a real distinguishing feature. Only fires for category
    terms actually present in _PRODUCT_TYPE_TERMS, so it can't drift as
    new, unrelated vocabulary gets used in queries."""
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
    """True if the query's words span 2+ DIFFERENT brand families (not
    variants of the same one) without the query naming any single
    product completely or naming at least one brand directly. Catches
    e.g. "size of H2 LD" -- "H2" and "LD" are too short/generic to
    register as distinctive or shared tokens for retrieval-scoping
    purposes (see _distinctive_product_tokens / _shared_family_tokens),
    but the phrase still genuinely spans multiple, materially different
    products (e.g. FIXaHY H2 LD XX, a fixed unit, vs. VISION H2 LD XX) --
    unlike "the FIXaHY sensor", which spans only variants of one brand,
    or "FIXaHY vs PORTaHY", which names both brands directly on purpose."""
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
        # A brand was named directly -- the cross-brand overlap above is
        # just incidental (other brands happening to share a generic
        # term). Check whether the query's tokens narrow to one clear
        # variant within the NAMED brand specifically, e.g. "FIXaHY H2
        # LD" (missing "XX") still matches FIXaHY H2 LD XX on 3 of its 4
        # tokens, versus just 1 (the bare brand name) for FIXaHY Analyzer
        # Series or FIXaHY-4220MA-RRNNVVII.
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
    """True if the query refers to a product generically ("this", "it",
    "the sensor"...) without naming a real one, OR spans multiple
    different product families via generic shared terms without naming
    any of them directly (see _cross_brand_ambiguous). Retrieval can't
    resolve either kind of reference on its own -- a topical/keyword
    match only means some excerpt discusses the same feature asked
    about, not that it's confirmed to be the product the user has in
    mind, so callers should ask for clarification instead of treating
    retrieved matches as an answer. A shared family prefix (e.g. "the
    FIXaHY sensor") still counts as naming something real, even though
    it doesn't narrow to one specific variant -- broad is not the same
    as empty. A "list all X" style query (see _is_list_all_products_query)
    is a different intent entirely -- deliberately asking for every
    matching product, not confused about which single one -- so it's
    never treated as ambiguous even when its wording spans multiple
    brands (e.g. "list all fixed H2 detectors"). A category term shared
    by multiple products (see _product_type_ambiguous) is also ambiguous
    even when it happens to be one product's literal name text (e.g.
    "the leak detector" naming AURIGA only by naming-convention accident,
    when 5 of 6 products are actually leak detectors)."""
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
    _is_single_product_document) rather than an actual product
    catalogue. Proven necessary by testing: asked "What is our hydrogen
    detection positioning?", retrieval returned 11/11 chunks from the
    competitor-comparison document and zero from any real product
    catalogue, and the model answered by describing a named competitor's
    (MSA) products as this company's own -- a prompt instruction telling
    it not to do this was tried first and reproducibly failed to
    prevent it (same class of problem as the ambiguous-product-reference
    and ungrounded-email checks: an attribution error, not a wording
    problem, so it needs a deterministic check on what was actually
    retrieved rather than another prompt instruction)."""
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
    """Reassembles a flat match list into round-robin order across
    product_names -- best-of-product-1, best-of-product-2, ..., then
    second-best-of-product-1, etc. -- instead of one contiguous block per
    product (which still buries every product but the first behind a
    wall of one product's content) or a plain similarity sort (which
    silently re-introduces the exact per-product domination
    _retrieve_balanced_across_products exists to prevent). Confirmed by
    testing: for "auriga vs fixahy", keyword-boost added 3 more chunks to
    AURIGA specifically (already the highest-scoring product), and a
    subsequent global sort pushed 2 of the 4 compared products' chunks to
    position 17+ of a 43-chunk context -- the LLM then only discussed the
    2 products it saw first."""
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
    """Retrieves the top_k most relevant chunks for a question, plus any
    keyword-matched chunks the vector search missed (see
    _keyword_boost_matches), along with a confidence level ("high", "low",
    or "none") based on similarity. exclude_document_names filters those
    documents out of the vector search itself (not just the returned
    results) -- for a caller that already knows some documents can never
    be a valid answer (e.g. discovery_generator.py excluding the sales-
    history spreadsheet from product recommendations), filtering after
    the fact isn't enough: proven by testing, a dominant non-product
    document can occupy the entire top-k regardless of how wide it's
    fetched, leaving nothing real behind after post-hoc filtering.

    scope_to_products=False skips narrowing to a single detected product
    entirely -- for callers that need breadth across multiple MNST
    products AND non-product documents (e.g. a competitor comparison) in
    the same call, like sales_aid_generator.py. Confirmed by testing: a
    comparison query naming both a use case and a competitor technology
    can still match one MNST product's distinctive tokens, which then
    silently scopes the ENTIRE top_k to that one product's chunks and
    excludes the competitor-comparison document completely.

    mentioned_products, when given, is used in place of this function's
    own _mentioned_products_for_query detection -- for a caller (the
    dispatcher) that already resolved which products the query names
    with better information than exact-token matching can (e.g. fuzzy
    typo tolerance: "porthay" for PORTaHY). Proven necessary by testing:
    even after dispatcher.py excluded every other product's document,
    this function's own re-detection still came back with only the one
    exact-matched product, which then filtered the surviving pool down
    to just that product's chunks by product_name -- silently undoing
    the caller's own, more complete answer instead of using it."""
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

    # mentioned_products is None checked first: when the caller (the
    # dispatcher) has already resolved specific products, that decision
    # must win over this function's own, older "list every document"
    # shortcut. Proven necessary by testing: "compare auriga and all
    # fixahy products" contains "all"/"products", which _is_list_all_
    # products_query reads as a request to list literally every document
    # in the whole collection -- silently discarding the dispatcher's
    # correct 4-product scoping and exclusion entirely.
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
            # Always interleave here, whether or not boosting added
            # anything -- _retrieve_balanced_across_products' own output
            # is already contiguous blocks (all of product A, then all of
            # product B, ...), which still buries every product but the
            # first behind a wall of one product's content. A plain
            # similarity sort would be even worse: it would silently undo
            # the fairness _retrieve_balanced_across_products just built
            # (confirmed by testing -- see _interleave_balanced_matches).
            matches = _interleave_balanced_matches(matches, mentioned)

            # Supplement with a genuinely unscoped pass (respecting only
            # exclude_where, not the product_name restriction) appended
            # after the balanced block, never mixed into the interleave
            # itself. Proven necessary by testing: dispatcher.py forces
            # mentioned_products to every currently approved product for
            # a question naming no specific one (e.g. "what industries
            # use the fixed hydrogen leak detector"), so that a spec
            # comparison gets fair per-product coverage -- but
            # _retrieve_balanced_across_products queries strictly by
            # product_name, which structurally can never surface a
            # reference document (the Industry Use Case Guide, exactly
            # what that question needs) since its chunks aren't tagged
            # with any of those product names at all. This has no effect
            # when exclude_where already rules reference documents out
            # (a genuine multi-product comparison, which intentionally
            # excludes them -- see dispatcher._scoped_result).
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
                pass  # the balanced, per-product matches already found are enough to proceed on
        elif boosted:
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
