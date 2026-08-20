"""Point 3: routes a question to whichever mechanism can answer it most
reliably, before falling through to unscoped semantic RAG (Path 1,
unchanged). Routing is deterministic pattern/alias matching, not LLM-
driven -- same principle as every other dispatch decision in this
codebase (is_ambiguous_product_reference, etc.): a routing mistake should
be debuggable and testable, not a probabilistic judgment call.

Three possible outcomes from dispatch(), checked most-specific first:

1. Decode (Path 3) / Catalog query (Path 2) -- a fully deterministic
   answer. Returned as (answer_text, retrieval_dict) where retrieval_dict
   is shaped exactly like retriever.retrieve()'s output, so it flows
   through response_generator.finalize_answer() completely unchanged --
   the same claim-checking, source display, and confidence handling as
   every other answer, not a separate code path to keep in sync. The LLM
   is never called for these.

2. Product resolution (Path 2) -- one or more products' aliases matched
   (including via fuzzy typo tolerance), but not a specific nomenclature
   code and not a catalog-filter question. Returned as a set of document
   names to EXCLUDE from retrieval (plus, for 2+ products, the resolved
   name list itself -- see DispatchResult.scoped_product_names), so
   semantic generation still happens normally -- an explanatory or
   comparison question genuinely needs prose, not a lookup -- just scoped
   to exactly the real, relevant documents instead of the whole corpus.
   Deliberately never blocks with a clarification question here, even
   when a name is ambiguous (e.g. bare "FIXaHY" spanning 3 products):
   every real candidate is already known, so scoping to all of them and
   letting citation-grounded generation answer per-product is strictly
   more useful than a dead-end question the rep has to respond to before
   getting anything.

3. None -- not a question this dispatcher can answer; the caller proceeds
   exactly as before.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Optional

from app.concentration import ConcentrationRange
from app.product_index import NomenclatureSegment, ProductIndexEntry, load_product_index
from app.registry_builder import INSTALL_TYPE_VOCAB, PRODUCT_TYPE_VOCAB, TECHNOLOGY_VOCAB

# Unconditionally safe to answer with the complete, unfiltered catalog --
# these phrasings are genuinely asking for everything, by construction,
# so there's no risk of mistaking a different question for this one.
_LIST_ALL_RE = re.compile(
    r"\b(list all|show (all\s+)?(portable|fixed)?\s*products|"
    r"what products (do you|does \w+) (offer|have|sell|make|provide|carry))\b",
    re.IGNORECASE,
)

# "which/what products ..." on its own is NOT safe to answer with the
# whole catalog when no real attribute matches: "which product is the
# lightest" and "which products are the cheapest/fastest" share this
# surface shape with a genuine filter question ("which products use
# SSEC"), but ask about a spec (weight, price, speed) the Product Index
# doesn't track at all. This phrasing only produces an answer when
# _match_all_attributes() actually finds something (see dispatch()) --
# with no match, it falls through to real search instead of a confidently
# wrong "here's everything" non-answer.
_WHICH_WHAT_PRODUCTS_RE = re.compile(
    r"\b(which|what) (products|sensors|detectors|devices|analyzers|units)\b",
    re.IGNORECASE,
)

# Weaker, more conversational phrasings ("I want to know about portable
# detectors") that don't name a filter as explicitly as _LIST_TRIGGER_RE's
# "which/what products" -- only trusted to trigger a catalog query when
# _match_attribute() also finds a real, bounded-vocabulary filter term, so
# a genuine explanatory question ("tell me about AURIGA") isn't misread as
# a request to dump the whole catalog.
_ASK_ABOUT_RE = re.compile(
    r"\b(tell me about|i want to know about|know about|information about|info about)\b",
    re.IGNORECASE,
)

# Same weak-trigger treatment as _ASK_ABOUT_RE, for bare "list"/"show"
# phrasings that don't say "list all" -- e.g. "list analyzers" names a
# real, tracked product_type value but neither _LIST_ALL_RE (needs the
# literal "list all") nor _WHICH_WHAT_PRODUCTS_RE (needs a "which/what"
# framing) catches it on its own, which without this would fall through
# to an unscoped, all-products retrieval instead of the deterministic
# Product Index answer. Only trusted alongside a real attribute match,
# same as _ASK_ABOUT_RE, so a bare "list"/"show" with no real filter
# still falls through to the existing safe behavior.
_LIST_KEYWORD_RE = re.compile(r"\b(list|show)\b", re.IGNORECASE)

# A question asking about "the [attribute] sensor/detector/..." as if it
# names exactly one product -- e.g. "the portable sensor" -- when the
# named attribute may actually match several real products. Reuses the
# same generic-noun set retriever.is_ambiguous_product_reference() already
# relies on for the same judgment call, just applied here so the Product
# Index can answer with the real candidate names instead of a generic
# "which product?" message.
_SINGULAR_REFERENCE_RE = re.compile(
    r"\bthe (?:\w+\s+){0,2}(sensor|detector|product|device|unit|instrument|system|analyzer)\b",
    re.IGNORECASE,
)

# Signals the query intends a comparison against something else -- used
# to make the single-match scoping branch more cautious: if the query
# also names something that DIDN'T resolve to a real product (a typo,
# e.g. "auriga vs porthay", or a genuinely new/unindexed product), hard-
# scoping to only the one recognized product would silently exclude the
# other side of the comparison entirely and produce a confident "I don't
# have that information" instead of an honest, broader search.
_COMPARISON_SIGNAL_RE = re.compile(
    r"\b(vs\.?|versus|compare[ds]?|comparison|difference between|compared to)\b",
    re.IGNORECASE,
)

_ATTRIBUTE_VOCAB = {"product_type": PRODUCT_TYPE_VOCAB, "install_type": INSTALL_TYPE_VOCAB, "technology": TECHNOLOGY_VOCAB}


@dataclass
class DispatchResult:
    kind: str  # "direct" (fully answered) or "scoped" (narrow retrieval, still generate)
    answer_text: Optional[str] = None
    retrieval: Optional[dict[str, Any]] = None
    exclude_document_names: Optional[set[str]] = None
    # Set only when 2+ products were resolved here (including via fuzzy
    # typo matching) -- passed through to retriever.retrieve()'s
    # mentioned_products so its own, separate (and typo-blind) product
    # detection doesn't override this dispatcher's more complete answer
    # and silently narrow back down to fewer products than were matched.
    scoped_product_names: Optional[list[str]] = None
    # Every scoped product's complete ordering-configuration table,
    # attached unconditionally rather than only when a question's wording
    # happens to trigger a lookup -- ordinary semantic retrieval doesn't
    # reliably rank the ordering-code chunk highly enough, and there's no
    # way to predict every phrasing ("configuration options", "XX values",
    # "variants", "models"...) a rep might use to ask for it. The data is
    # already known and cheap to include, so it's always included.
    nomenclature_notes: Optional[list[dict[str, str]]] = None


def _contains_term(term: str, text: str) -> bool:
    """Whole-term match, robust to a term ending or starting with a
    non-word character (e.g. alias "Auriga 10%") -- plain \\b alone fails
    right after a symbol at the very end of a string, since \\b needs a
    transition between a word and non-word character, and a symbol
    followed by nothing is non-word on both sides. Negative lookaround on
    alphanumerics doesn't have that blind spot."""
    return re.search(r"(?<![a-zA-Z0-9])" + re.escape(term.lower()) + r"(?![a-zA-Z0-9])", text) is not None


def _match_attribute(query: str, tech_aliases: dict[str, str]) -> Optional[tuple[str, str]]:
    """Finds a (field, value) filter the query names -- e.g. "portable" ->
    ("install_type", "Portable"), "SSEC" -> ("technology", "Solid State
    Electrochemical") via the human-supplied alias table (see
    registry_builder.TECHNOLOGY_ALIASES for why that one can't be
    extracted from the documents)."""
    query_lower = query.lower()
    for alias, canonical in sorted(tech_aliases.items(), key=lambda kv: len(kv[0]), reverse=True):
        if _contains_term(alias, query_lower):
            return "technology", canonical
    for field, vocab in _ATTRIBUTE_VOCAB.items():
        for term in sorted(vocab, key=len, reverse=True):
            if term.lower() in query_lower:
                return field, term
    return None


def _match_all_attributes(query: str, tech_aliases: dict[str, str]) -> list[tuple[str, str]]:
    """Every (field, value) filter the query names, not just the first --
    e.g. "list all SSEC and MEMS products" names two technology values at
    once. _match_attribute stops at the first hit (fine for its own
    yes/no "does this name an attribute at all" use elsewhere), which
    would otherwise understate a "list all X and Y" query down to just
    X."""
    query_lower = query.lower()
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for alias, canonical in sorted(tech_aliases.items(), key=lambda kv: len(kv[0]), reverse=True):
        if _contains_term(alias, query_lower):
            key = ("technology", canonical)
            if key not in seen:
                seen.add(key)
                found.append(key)
    for field, vocab in _ATTRIBUTE_VOCAB.items():
        for term in sorted(vocab, key=len, reverse=True):
            if term.lower() in query_lower:
                key = (field, term)
                if key not in seen:
                    seen.add(key)
                    found.append(key)
    return found


def _synthetic_retrieval(source_catalogues: set[str], answer_text: str) -> dict[str, Any]:
    """A retrieval-shaped dict so a deterministic answer flows through
    finalize_answer() unchanged."""
    names = sorted(source_catalogues) or ["Product Registry"]
    matches = [
        {"text": answer_text, "similarity": 1.0, "metadata": {"document_name": n, "product_name": n}}
        for n in names
    ]
    return {"matches": matches, "confidence": "high"}


def _filter_by_attribute(products: list[ProductIndexEntry], field: str, value: str) -> list[ProductIndexEntry]:
    """Products matching a (field, value) filter from _match_attribute().
    Technology uses containment, not equality: the vocab deliberately
    holds both a specific term ("CMOS MEMS") and its broader category
    ("MEMS") as distinct entries (see registry_builder.TECHNOLOGY_VOCAB),
    so a query for "MEMS products" -- which resolves to the "MEMS" vocab
    entry -- must still match a product whose extracted technology is the
    more specific "CMOS MEMS", not just a product whose field is the bare
    string "MEMS" verbatim. install_type/product_type have no such
    specific/general split today, so exact equality there is unaffected."""
    if field == "technology":
        return [p for p in products if value.lower() in getattr(p, field).lower()]
    return [p for p in products if getattr(p, field) == value]


def _nomenclature_summary(product: ProductIndexEntry) -> Optional[str]:
    """Renders a product's complete, decoded ordering-nomenclature table
    as plain text -- every code it can be ordered with, read directly
    from the registry, never guessed. See DispatchResult.nomenclature_notes
    for why this is always attached rather than conditionally triggered."""
    if not isinstance(product.nomenclature, dict):
        return None
    lines = []
    for segment in product.nomenclature.values():
        if not isinstance(segment, NomenclatureSegment) or not segment.values:
            continue
        parts = []
        for code, value in segment.values.items():
            text = str(value) if isinstance(value, ConcentrationRange) else value
            parts.append(f"{code} = {text}")
        if parts:
            lines.append(f"{segment.label}: " + "; ".join(parts))
    if not lines:
        return None
    return f"{product.product_name} ordering configuration -- " + " | ".join(lines)


def _reference_document_names() -> set[str]:
    """Every currently-indexed document that ISN'T a single-product
    catalogue -- the competitor comparison, historical sales log, and
    cross-category use-case guide. Reuses retriever._is_single_product_
    document, the same rule the vector index itself already applies for
    this distinction, so it can't drift out of sync with it. Returns an
    empty set (never crashes a dispatch decision) if the index isn't
    reachable for any reason."""
    from app.retriever import _is_single_product_document, get_collection

    try:
        collection = get_collection()
        names = {m.get("document_name", "") for m in collection.get(include=["metadatas"])["metadatas"]}
    except Exception:
        return set()
    return {n for n in names if n and not _is_single_product_document(n)}


def _scoped_result(
    all_products: list[ProductIndexEntry],
    involved: list[ProductIndexEntry],
    exclude_references: bool = True,
) -> "DispatchResult":
    """Builds a "scoped" DispatchResult for one or more resolved
    products, consistently -- one place that always attaches both
    scoped_product_names (2+ products) and nomenclature_notes, so a
    scoping decision can't accidentally skip either the way four separate
    hand-written return statements already once did.

    exclude_references=True (the default) also excludes every reference
    document (competitor comparison, historical sales, use-case guide),
    since none of them are in the Product Registry and so were never in
    the "other catalogues to exclude" set on their own. A question naming
    one or more specific products has no legitimate reason to draw on a
    cross-vendor comparison sheet.
    The one caller that wants reference docs to stay eligible -- no
    specific product named at all, balancing across the whole catalog --
    passes exclude_references=False explicitly."""
    keep = {p.source_catalogue for p in involved}
    exclude = {p.source_catalogue for p in all_products if p.source_catalogue not in keep}
    if exclude_references:
        exclude |= _reference_document_names()
    notes = [
        {"document_name": p.source_catalogue, "product_name": p.product_name, "text": summary}
        for p in involved
        for summary in [_nomenclature_summary(p)]
        if summary
    ]
    return DispatchResult(
        kind="scoped",
        exclude_document_names=exclude,
        scoped_product_names=[p.product_name for p in involved] if len(involved) >= 2 else None,
        nomenclature_notes=notes or None,
    )


def _answer_catalog_query(
    query: str, products: list[ProductIndexEntry], tech_aliases: dict[str, str]
) -> tuple[str, dict[str, Any]]:
    """Path 2 -- Catalog Queries. Filters the Product Index deterministically
    and states the complete, exact matching set directly -- never a second
    retrieval pass, so a real match can't be silently dropped by top-k
    truncation the way plain-RAG enumeration would."""
    matches = _match_all_attributes(query, tech_aliases)
    if matches:
        matching_by_name: dict[str, ProductIndexEntry] = {}
        for field, value in matches:
            for p in _filter_by_attribute(products, field, value):
                matching_by_name[p.product_name] = p
        matching = sorted(matching_by_name.values(), key=lambda p: p.product_name)
        by_field: dict[str, list[str]] = {}
        for field, value in matches:
            by_field.setdefault(field, []).append(value)
        label = " and ".join(
            f'{field.replace("_", " ")} "{" or ".join(values)}"' for field, values in by_field.items()
        )
    else:
        matching = list(products)
        label = "the approved catalog"

    if not matching:
        answer = f"No approved products are documented with {label}."
    else:
        names = ", ".join(p.product_name for p in matching)
        answer = f"{len(matching)} product(s) match {label}: {names}."

    return answer, _synthetic_retrieval({p.source_catalogue for p in matching}, answer)


_MIN_FUZZY_WORD_LEN = 4
_FUZZY_RATIO_THRESHOLD = 0.75


def _fuzzy_brand_match(
    query_words: list[str], products: list[ProductIndexEntry], already_matched: set[str]
) -> list[tuple[ProductIndexEntry, str]]:
    """Typo tolerance for a brand name exact alias matching missed
    entirely -- e.g. "porthay" for PORTaHY. Compares whole query words
    against each product's own brand-root token (the first word of its
    product name) using edit-distance ratio, with a high threshold and a
    near-matching length so a short/generic word can't accidentally
    fuzzy-match a brand name (checked: "range"/"auriga" scores 0.36,
    "vision"/"fixahy" scores 0.17, both safely below threshold; the real
    typo "porthay"/"portahy" scores 0.86). Deterministic and testable,
    not a guess -- ordinary typo tolerance, the same category as already
    matching "FIXaHY"/"fixahy" case-insensitively. Only considered for a
    product not already found by exact matching, so it never overrides a
    real match."""
    found: list[tuple[ProductIndexEntry, str]] = []
    for p in products:
        if p.product_name in already_matched:
            continue
        brand = p.product_name.split()[0] if p.product_name else ""
        if not brand:
            continue
        for word in query_words:
            if len(word) < _MIN_FUZZY_WORD_LEN or abs(len(word) - len(brand)) > 2:
                continue
            if word.lower() == brand.lower():
                continue  # exact match would already have caught this
            ratio = difflib.SequenceMatcher(None, word.lower(), brand.lower()).ratio()
            if ratio >= _FUZZY_RATIO_THRESHOLD:
                found.append((p, brand))
                break
    return found


def _matching_products(query: str, products: list[ProductIndexEntry]) -> list[tuple[ProductIndexEntry, str]]:
    """Every product with at least one alias appearing in the query,
    paired with its longest matching alias -- excluding a match whose
    alias is wholly subsumed by a different product's longer match (e.g.
    the bare brand root "FIXaHY" also matching the Analyzer and 4220MA
    when the query actually said the more specific "FIXaHY H2 LD V").
    Without this, the bare brand-root alias (needed so "PORTaHY" alone
    resolves, and so a genuine multi-product comparison isn't miscounted
    as one product) would dilute a single, fully-coded product mention
    into several matches instead of resolving cleanly to the one
    actually named.

    Falls back to fuzzy brand matching (_fuzzy_brand_match) for any product
    no exact alias caught -- a typo like "porthay" for PORTaHY would
    otherwise match nothing, and the caller's "exactly one match" scoping
    logic would hard-exclude that product's real catalogue from retrieval
    entirely, producing a confident "no information" answer instead of an
    honest comparison."""
    query_lower = query.lower()
    found: list[tuple[ProductIndexEntry, str]] = []
    for p in products:
        best_alias = None
        for alias in p.aliases:
            if _contains_term(alias, query_lower):
                if best_alias is None or len(alias) > len(best_alias):
                    best_alias = alias
        if best_alias is not None:
            found.append((p, best_alias))

    already_matched = {p.product_name for p, _ in found}
    query_words = re.findall(r"[a-zA-Z0-9]+", query)
    found.extend(_fuzzy_brand_match(query_words, products, already_matched))

    all_aliases = [alias.lower() for _, alias in found]
    return [
        (p, alias) for p, alias in found
        if not any(alias.lower() != other and alias.lower() in other for other in all_aliases)
    ]


def _ambiguous_alias_groups(
    matches: list[tuple[ProductIndexEntry, str]]
) -> dict[str, list[ProductIndexEntry]]:
    """Groups matched products by the literal alias text that matched them
    -- e.g. a bare "FIXaHY" mention matching three distinct real products
    that share that brand root. Grouped by the matched alias itself, not
    the registry's "family" field: a product whose product_name has no
    space (FIXaHY-G/P/E-4220MA-RRNNVVII) reports its own full name as its
    family and would never group with its siblings that way. Only a
    group with 2+ different products is ambiguous -- a family referenced
    by a single member (e.g. "PORTaHY" alone, which only ever matches
    PORTaHY H2 LD XX) is already fully resolved."""
    by_alias: dict[str, list[ProductIndexEntry]] = {}
    for p, alias in matches:
        by_alias.setdefault(alias.lower(), []).append(p)
    return {alias: members for alias, members in by_alias.items() if len(members) > 1}


def _answer_scoped_family_list(ambiguous: dict[str, list[ProductIndexEntry]]) -> tuple[str, dict[str, Any]]:
    """Path 2 -- Catalog Query, scoped to a specifically-named but
    multi-member family (e.g. "list all the products of the FIXaHY
    family"). The query already said "list"/"all" -- that's a deliberate
    request to enumerate, not confusion about which one is meant -- so
    this states exactly that family's real members, not a clarification
    request (_answer_family_ambiguity, for a non-list query like "AURIGA
    vs FIXaHY") and not the whole catalog (_answer_catalog_query would
    ignore "FIXaHY" entirely, since it's not a technology/install-type/
    product-type term)."""
    seen: dict[str, ProductIndexEntry] = {}
    for members in ambiguous.values():
        for p in members:
            seen[p.product_name] = p
    ordered = sorted(seen.values(), key=lambda p: p.product_name)
    names = ", ".join(p.product_name for p in ordered)
    label = " / ".join(sorted(ambiguous.keys()))
    answer = f'{len(ordered)} product(s) in the "{label}" family: {names}.'
    return answer, _synthetic_retrieval({p.source_catalogue for p in ordered}, answer)


def _matched_code(product: ProductIndexEntry, alias: str) -> Optional[str]:
    """If the matched alias ends in a specific nomenclature code (e.g.
    "FIXaHY H2 LD V" ending in "V"), returns that code -- not just that
    the product was named, but which of its decoded values specifically."""
    if not isinstance(product.nomenclature, dict):
        return None
    for segment in product.nomenclature.values():
        if isinstance(segment, NomenclatureSegment):
            for code in segment.values:
                if alias.lower() == f"{alias.rsplit(' ', 1)[0]} {code}".lower() and alias.lower().endswith(code.lower()):
                    return code
    return None


# A question naming one specific field (e.g. "portahy h2 ld 5k what is
# the resolution") should get just that field, not the whole segment
# description -- checked against registry_builder._parse_selectable_
# table's own "Field: value; Field: value; ..." shape, so this can't
# drift out of sync with how that data is actually built.
_FIELD_KEYWORDS = {
    "resolution": "Resolution",
    "minimum detection limit": "MDL",
    "detection limit": "MDL",
    "mdl": "MDL",
    "accuracy": "Accuracy",
    "span": "Span",
    "start": "Start",
}


def _extract_field(value_text: str, field_label: str) -> Optional[str]:
    """Pulls one specific field's value out of a "Field: value; Field:
    value; ..." structured description -- e.g. asked only for
    "resolution", returns just "5 ppm" instead of the whole Start/Span/
    Resolution/MDL/Accuracy block. None if value_text isn't structured
    this way (e.g. a plain Output Signal meaning like "Analogue Voltage
    Output" has no "Field: value" shape to extract from) or doesn't
    contain the requested field."""
    for part in value_text.split(";"):
        part = part.strip()
        if ":" not in part:
            continue
        label, _, val = part.partition(":")
        if field_label.lower() in label.strip().lower():
            return val.strip()
    return None


def _answer_decode_query(
    product: ProductIndexEntry, code: str, question: str = ""
) -> Optional[tuple[str, dict[str, Any]]]:
    """Path 3 -- direct decode of a specific nomenclature code, e.g. "V"
    -> "Analogue Voltage Output" for FIXaHY H2 LD XX. Read directly from
    the registry, never computed or guessed."""
    if not isinstance(product.nomenclature, dict):
        return None
    question_lower = question.lower()
    requested_field = next(
        (label for keyword, label in _FIELD_KEYWORDS.items() if keyword in question_lower), None
    )
    for segment in product.nomenclature.values():
        if isinstance(segment, NomenclatureSegment) and code in segment.values:
            value = segment.values[code]
            value_text = str(value) if isinstance(value, ConcentrationRange) else value

            if requested_field and isinstance(value, str):
                extracted = _extract_field(value_text, requested_field)
                if extracted is not None:
                    answer = f'For the {product.product_name}, code "{code}" -- {requested_field}: {extracted}.'
                    return answer, _synthetic_retrieval({product.source_catalogue}, answer)

            answer = (
                f'For the {product.product_name}, code "{code}" in the {segment.label} segment '
                f"means: {value_text}."
            )
            return answer, _synthetic_retrieval({product.source_catalogue}, answer)
    return None


def dispatch(question: str) -> Optional[DispatchResult]:
    """Entry point. Returns None if this question should go through
    normal, unscoped semantic RAG exactly as before."""
    try:
        products, tech_aliases = load_product_index()
    except (FileNotFoundError, OSError):
        return None  # registry not built yet -- fall through to normal RAG

    matches = _matching_products(question, products)

    # Exactly one product named: check for a specific decoded code first
    # (more specific than a bare product mention), then fall back to
    # scoping retrieval to that one product.
    if len(matches) == 1:
        product, alias = matches[0]
        code = _matched_code(product, alias)
        if code is not None:
            result = _answer_decode_query(product, code, question)
            if result is not None:
                answer_text, retrieval = result
                return DispatchResult(kind="direct", answer_text=answer_text, retrieval=retrieval)

        # A comparison signal ("vs", "compare", ...) alongside only one
        # recognized product means the query likely names a second thing
        # that just didn't resolve (a typo, or a product not yet in the
        # registry) -- scoping to only the one recognized product would
        # silently hide the other side of the comparison and produce a
        # confident wrong answer instead of an honest broader search.
        looks_like_catalog_question = bool(_LIST_ALL_RE.search(question)) or bool(_WHICH_WHAT_PRODUCTS_RE.search(question))
        if not looks_like_catalog_question and not _COMPARISON_SIGNAL_RE.search(question):
            return _scoped_result(products, [product])

    # An explicit "list"/"all" request is unconditionally a genuine
    # enumeration ask. "which/what products ..." and the weaker "tell me
    # about" phrasing both only count as a deliberate enumeration request
    # when a real attribute was actually found -- otherwise they're
    # equally likely to be a comparison/superlative question about a
    # spec the Product Index doesn't track (see _WHICH_WHAT_PRODUCTS_RE),
    # and must fall through to real search rather than a fabricated
    # "here's the whole catalog" answer. Computed once so every branch
    # below agrees on what counts as a deliberate enumeration request.
    attribute_matches = _match_all_attributes(question, tech_aliases)
    is_list_request = bool(_LIST_ALL_RE.search(question)) or (
        bool(attribute_matches)
        and (
            bool(_WHICH_WHAT_PRODUCTS_RE.search(question))
            or bool(_ASK_ABOUT_RE.search(question))
            or bool(_LIST_KEYWORD_RE.search(question))
        )
    )

    # Two or more matches: before falling through to unscoped RAG (which
    # would blend distinct products together), decide how to scope
    # rather than whether to answer at all -- a blocking clarification
    # question is worse than a properly-separated answer almost every
    # time: the rep can still narrow down with a follow-up, but a dead
    # end wastes their turn for no benefit, since every real candidate is
    # already known and generation is already grounded per-product by
    # citation -- a bare brand root like "FIXaHY" spanning 3 real products
    # shouldn't force a clarification when the query is otherwise
    # unambiguous. An explicit "list all X"
    # style request (is_list_request) is the one exception: a deliberate
    # enumeration should get the deterministic, complete name list, not
    # LLM prose.
    if len(matches) >= 2:
        ambiguous = _ambiguous_alias_groups(matches)
        if ambiguous and is_list_request:
            answer_text, retrieval = _answer_scoped_family_list(ambiguous)
            return DispatchResult(kind="direct", answer_text=answer_text, retrieval=retrieval)

        if not is_list_request:
            return _scoped_result(products, [p for p, _ in matches])

    if is_list_request:
        answer_text, retrieval = _answer_catalog_query(question, products, tech_aliases)
        return DispatchResult(kind="direct", answer_text=answer_text, retrieval=retrieval)

    # No product named at all, but a singular reference to an attribute
    # that could describe more than one real product -- e.g. "the
    # portable sensor" when both AURIGA and PORTaHY are Portable. Scope
    # to every matching candidate and let generation answer for each,
    # rather than blocking with a clarification question, for the same
    # reason as the multi-match case above.
    if not matches and _SINGULAR_REFERENCE_RE.search(question):
        match = _match_attribute(question, tech_aliases)
        if match:
            candidates = _filter_by_attribute(products, *match)
            if candidates:
                return _scoped_result(products, candidates)

    # No specific product named at all, and no registry attribute
    # narrowed the field either -- don't leave this to an ordinary,
    # un-balanced top-k search. Deliberately NOT gated on any particular
    # phrasing (superlative words, "the X sensor", ...): that was tried
    # and kept missing real cases, because there is no bounded list of
    # ways a question can implicitly span the whole catalog -- "smallest
    # product MNST sells", "which is the lightest product", and "what is
    # the recovery time after detecting a leak" all have completely
    # different surface shapes but share the same failure: plain top-k
    # search silently favors whichever product's chunks happen to rank
    # highest and answers as if that were checked against every product
    # (e.g. confidently describing AURIGA's recovery time and adding
    # "other products do not have specific recovery times documented" --
    # untrue; they were simply never retrieved). The one real cost is
    # that a genuinely product-agnostic question (no product implied at
    # all) also gets this treatment; that's a worthwhile trade against
    # silently wrong "comprehensive" answers.
    #
    # References stay eligible here EXCEPT for a self-referential
    # ("our"/"we"/"us"/"the company") question -- those still exclude
    # them, same as the named-product case. Checked directly with a live
    # LLM call: "What is our hydrogen detection positioning?" correctly
    # described only MNST's own products even with competitor content
    # available (guaranteed real product coverage now crowds it out) --
    # but the sources panel still showed 5 Competitor Comparison chunks
    # for a question about "our" positioning, which is confusing even
    # when unused, and the one check built specifically to prevent
    # competitor content from answering a self-referential question
    # (is_self_referential_without_own_products) can now never fire here
    # again, since real product matches are always guaranteed present.
    # Excluding references for this case keeps that original protection
    # meaningful instead of quietly making it unreachable.
    from app.retriever import _SELF_REFERENTIAL_RE

    if not matches:
        exclude_refs = bool(_SELF_REFERENTIAL_RE.search(question))
        return _scoped_result(products, products, exclude_references=exclude_refs)

    return None
