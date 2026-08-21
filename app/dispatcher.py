"""Routes a question to whichever mechanism can answer it most reliably,
before falling through to unscoped semantic RAG. Routing is deterministic
pattern/alias matching, not LLM-driven, so a routing mistake is
debuggable and testable rather than a probabilistic judgment call.

dispatch() returns one of three outcomes, checked most-specific first:

1. Decode / catalog query -- a fully deterministic answer, returned as
   (answer_text, retrieval_dict) shaped exactly like retriever.retrieve()'s
   output so it flows through response_generator.finalize_answer()
   unchanged. The LLM is never called for these.

2. Product resolution -- one or more products' aliases matched (including
   via fuzzy typo tolerance) but no specific nomenclature code or catalog
   filter. Returns document names to EXCLUDE from retrieval so semantic
   generation still runs, just scoped to the relevant documents. Never
   blocks with a clarification question even for an ambiguous name (e.g.
   bare "FIXaHY" spanning 3 products) -- scoping to every candidate and
   letting grounded generation answer per-product beats a dead end.

3. None -- not answerable here; the caller proceeds as before.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Optional

from app.concentration import ConcentrationRange
from app.product_index import NomenclatureSegment, ProductIndexEntry, load_product_index
from app.registry_builder import INSTALL_TYPE_VOCAB, PRODUCT_TYPE_VOCAB, TECHNOLOGY_VOCAB

# Unconditionally safe to answer with the whole catalog -- these phrasings
# are genuinely asking for everything, by construction.
_LIST_ALL_RE = re.compile(
    r"\b(list all|show (all\s+)?(portable|fixed)?\s*products|"
    r"what products (do you|does \w+) (offer|have|sell|make|provide|carry))\b",
    re.IGNORECASE,
)

# NOT safe to answer with the whole catalog on its own: "which product is
# the lightest" shares this shape with a real filter question ("which
# products use SSEC") but asks about a spec the Product Index doesn't
# track. Only produces an answer when _match_all_attributes() finds
# something; otherwise falls through to real search.
_WHICH_WHAT_PRODUCTS_RE = re.compile(
    r"\b(which|what) (products|sensors|detectors|devices|analyzers|units)\b",
    re.IGNORECASE,
)

# Weaker, conversational phrasing ("I want to know about portable
# detectors") -- only trusted alongside a real _match_attribute() filter
# term, so "tell me about AURIGA" isn't misread as a catalog dump.
_ASK_ABOUT_RE = re.compile(
    r"\b(tell me about|i want to know about|know about|information about|info about)\b",
    re.IGNORECASE,
)

# Same weak-trigger treatment as _ASK_ABOUT_RE, for bare "list"/"show"
# phrasings _LIST_ALL_RE and _WHICH_WHAT_PRODUCTS_RE don't catch (e.g.
# "list analyzers") -- only trusted alongside a real attribute match.
_LIST_KEYWORD_RE = re.compile(r"\b(list|show)\b", re.IGNORECASE)

# "the [attribute] sensor/detector/..." phrased as if naming one product
# (e.g. "the portable sensor") when it may match several. Reuses the same
# generic-noun set retriever.is_ambiguous_product_reference() relies on,
# so the Product Index can answer with real candidate names instead of a
# generic "which product?" message.
_SINGULAR_REFERENCE_RE = re.compile(
    r"\bthe (?:\w+\s+){0,2}(sensor|detector|product|device|unit|instrument|system|analyzer)\b",
    re.IGNORECASE,
)

# Signals a comparison intent, so the single-match scoping branch can stay
# cautious: if the query also names something that didn't resolve (a
# typo, e.g. "auriga vs porthay"), hard-scoping to the one recognized
# product would silently drop the other side of the comparison.
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
    # typo matching) -- passed to retriever.retrieve()'s mentioned_products
    # so its own typo-blind detection doesn't narrow back down.
    scoped_product_names: Optional[list[str]] = None
    # Every scoped product's ordering-configuration table, attached
    # unconditionally -- semantic retrieval doesn't reliably rank the
    # ordering-code chunk, and there's no way to predict every phrasing
    # ("configuration options", "XX values", "variants") a rep might use.
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
    e.g. "list all SSEC and MEMS products" names two at once, unlike
    _match_attribute which stops at the first hit."""
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
    Technology uses containment, not equality: the vocab holds both a
    specific term ("CMOS MEMS") and its broader category ("MEMS") as
    distinct entries, so "MEMS products" must still match a product whose
    technology is "CMOS MEMS". install_type/product_type use equality."""
    if field == "technology":
        return [p for p in products if value.lower() in getattr(p, field).lower()]
    return [p for p in products if getattr(p, field) == value]


def _nomenclature_summary(product: ProductIndexEntry) -> Optional[str]:
    """Renders a product's decoded ordering-nomenclature table as plain
    text, read directly from the registry, never guessed."""
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
    """Every indexed document that isn't a single-product catalogue (the
    competitor comparison, historical sales log, use-case guide). Reuses
    retriever._is_single_product_document so it can't drift out of sync.
    Returns an empty set rather than raising if the index is unreachable."""
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
    """Builds a "scoped" DispatchResult for one or more resolved products
    -- the one place that attaches both scoped_product_names (2+ products)
    and nomenclature_notes, so a scoping decision can't skip either.

    exclude_references=True (default) also excludes reference documents
    (competitor comparison, historical sales, use-case guide), since a
    question naming specific products has no reason to draw on those. The
    one caller balancing across the whole catalog with no product named
    passes exclude_references=False."""
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
    entirely -- e.g. "porthay" for PORTaHY. Compares query words against
    each product's brand-root token (first word of the name) via
    edit-distance ratio, with a high threshold and near-matching length
    so a generic word can't false-positive ("range"/"auriga" scores 0.36,
    real typo "porthay"/"portahy" scores 0.86). Only considered for a
    product exact matching missed, so it never overrides a real match."""
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
    """Every product with at least one alias in the query, paired with its
    longest matching alias -- excludes a match subsumed by a different
    product's longer match (e.g. bare "FIXaHY" also matching the Analyzer
    and 4220MA when the query said the more specific "FIXaHY H2 LD V"),
    so a single fully-coded mention resolves to one product, not several.

    Falls back to fuzzy brand matching (_fuzzy_brand_match) for any
    product no exact alias caught, so a typo doesn't get hard-excluded
    from retrieval as if the product were genuinely unmentioned."""
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
    -- e.g. bare "FIXaHY" matching three distinct products sharing that
    brand root. Grouped by the alias itself, not the registry's "family"
    field, since a product with no space in its name would never group
    with siblings that way. Only a group with 2+ products is ambiguous."""
    by_alias: dict[str, list[ProductIndexEntry]] = {}
    for p, alias in matches:
        by_alias.setdefault(alias.lower(), []).append(p)
    return {alias: members for alias, members in by_alias.items() if len(members) > 1}


def _answer_scoped_family_list(ambiguous: dict[str, list[ProductIndexEntry]]) -> tuple[str, dict[str, Any]]:
    """Catalog query scoped to a specifically-named, multi-member family
    (e.g. "list all the products of the FIXaHY family"). The query already
    said "list"/"all", so this states the family's real members directly
    -- not a clarification request (that's for a non-list query like
    "AURIGA vs FIXaHY") and not the whole catalog."""
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


# A question naming one specific field (e.g. "...what is the resolution")
# should get just that field, not the whole segment description. Matches
# registry_builder._parse_selectable_table's "Field: value; ..." shape.
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
    """Pulls one field's value out of a "Field: value; Field: value; ..."
    description, e.g. just "5 ppm" for "resolution" instead of the whole
    block. None if value_text isn't structured this way, or lacks the field."""
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

        # A comparison signal alongside only one recognized product likely
        # means a second product didn't resolve (typo, or not yet
        # registered) -- scoping to just the one would hide it silently.
        looks_like_catalog_question = bool(_LIST_ALL_RE.search(question)) or bool(_WHICH_WHAT_PRODUCTS_RE.search(question))
        if not looks_like_catalog_question and not _COMPARISON_SIGNAL_RE.search(question):
            return _scoped_result(products, [product])

    # An explicit "list"/"all" request is unconditionally genuine.
    # "which/what products" and "tell me about" only count as enumeration
    # requests when a real attribute was found -- otherwise they're
    # equally likely a comparison/superlative question about a spec the
    # Product Index doesn't track, and must fall through to real search.
    attribute_matches = _match_all_attributes(question, tech_aliases)
    is_list_request = bool(_LIST_ALL_RE.search(question)) or (
        bool(attribute_matches)
        and (
            bool(_WHICH_WHAT_PRODUCTS_RE.search(question))
            or bool(_ASK_ABOUT_RE.search(question))
            or bool(_LIST_KEYWORD_RE.search(question))
        )
    )

    # Two or more matches: scope rather than block with a clarification --
    # every real candidate is already known and generation is grounded
    # per-product by citation, so a bare brand root like "FIXaHY" spanning
    # 3 products shouldn't force a dead-end question. is_list_request is
    # the one exception: a deliberate enumeration gets the deterministic
    # name list, not LLM prose.
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

    # No product named, but a singular reference to an attribute that
    # could describe more than one product (e.g. "the portable sensor"
    # when AURIGA and PORTaHY are both Portable) -- scope to every
    # candidate, same reasoning as the multi-match case above.
    if not matches and _SINGULAR_REFERENCE_RE.search(question):
        match = _match_attribute(question, tech_aliases)
        if match:
            candidates = _filter_by_attribute(products, *match)
            if candidates:
                return _scoped_result(products, candidates)

    # No product named and no registry attribute narrowed the field --
    # don't leave this to an unbalanced top-k search. Deliberately not
    # gated on phrasing (superlatives, "the X sensor", ...): there's no
    # bounded list of ways a question can implicitly span the whole
    # catalog ("smallest product MNST sells", "recovery time after
    # detecting a leak"), and plain top-k search would silently favor
    # whichever product's chunks rank highest, then answer as if every
    # product were checked (e.g. describing only AURIGA's recovery time
    # while claiming others have none documented -- they were simply
    # never retrieved).
    #
    # References stay eligible here except for a self-referential
    # ("our"/"we"/"us"/"the company") question, which excludes them --
    # otherwise competitor content stays in the sources panel even when
    # unused, and is_self_referential_without_own_products (built to
    # block competitor content from answering "our" questions) could
    # never fire here since real product matches are now guaranteed.
    from app.retriever import _SELF_REFERENTIAL_RE

    if not matches:
        exclude_refs = bool(_SELF_REFERENTIAL_RE.search(question))
        return _scoped_result(products, products, exclude_references=exclude_refs)

    return None
