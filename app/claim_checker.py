#python -m app.claim_checker
"""Implements the "Claim checker" component (spec 4.1) and the claim-
checking rules in Implementation Details 12.3: scans a question and its
draft answer for restricted-claim terms (see data/restricted_claims.md),
decides which risk category applies, and blocks customer-facing wording
when a matched term isn't backed by the actual retrieved source text.

Deliberately keyword-based, not an LLM/ML classifier -- spec 12.3: "Keep
this simple initially using keyword rules; ML classification can be a
later improvement." Same reasoning as intent.py's classifier.

Risk categories match streamlit_app.py's RISK_COLORS keys exactly: None,
Certification, Accuracy, Safety, Pricing, Legal, Delivery. Kept in sync by
hand with data/restricted_claims.md -- update both together if the policy
changes.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

NONE_CATEGORY = "None"

# Checked in this order -- first category with a match wins, most specific/
# highest-stakes first (mirrors intent.py's priority-order rationale). A
# question mentioning both a named certification and a price shouldn't get
# labelled "Pricing" just because pricing happens to be checked last.
RESTRICTED_TERM_CATEGORIES: dict[str, list[str]] = {
    "Certification": [
        # Named standards/marks the docs explicitly do NOT hold (spec 8.2,
        # data/restricted_claims.md) -- these must never be confirmed. See
        # ALWAYS_UNSUPPORTED_TERMS below: presence in source text alone
        # isn't trusted for these, since the approved deck's own "IECEx &
        # ATEX in Process" line genuinely contains the words "atex" and
        # "iecex" while denying, not confirming, the claim.
        "atex", "iecex", "sil", "sil-2", "sil2",
        "life-safety", "life safety", "explosion proof", "explosion-proof",
        # "ce marked" listed separately -- the trailing-plural regex in
        # _contains_term doesn't cover "-ed", so "mark" alone wouldn't
        # catch the natural phrasing "Is this CE marked?".
        "ce mark", "ce marking", "ce marked",
        # Genuinely held certifications too -- a question about these is
        # still worth flagging, since the approval is scoped to a specific
        # product/configuration (e.g. PESO only covers the FIXaHY, Gas
        # Group IIC Zone 1) and must not be generalized to the whole line.
        # Unlike the ones above, these CAN legitimately be source-supported
        # for the right product, so they stay normally evidence-checked.
        "peso", "arai", "iec",
        "flameproof", "zone 1", "zone 2", "gas group", "ex ia",
        "intrinsically safe",
        "certified", "certification", "approved", "approval",
        "hazardous area", "hazardous zone",
    ],
    "Accuracy": [
        "ppm accuracy", "accuracy", "universal gas detection",
        "universal detection", "detects any gas", "100% accurate",
        "always accurate",
        # The MEMS platform's roadmap gas list (data/restricted_claims.md)
        # -- singular so the trailing-plural regex in _contains_term also
        # catches "hydrocarbons"/"refrigerants". Also in
        # ALWAYS_UNSUPPORTED_TERMS: the roadmap slide genuinely names these
        # gases, but that's the technology's target capability, not any
        # current product's spec.
        "helium", "methane", "sf6", "hydrocarbon", "refrigerant",
    ],
    "Safety": [
        "safe to use", "completely safe", "guaranteed safe",
        "fail-safe", "failsafe", "no risk of",
    ],
    "Pricing": [
        "price", "pricing", "quotation", "quote", "cost", "discount",
        "warranty",
    ],
    "Delivery": [
        "delivery", "lead time", "guaranteed delivery",
        "ship date", "shipping date",
    ],
    "Legal": [
        "legal", "regulatory", "liability", "indemnif",
        "compliance guarantee",
    ],
}

# data/restricted_claims.md phrases these as "NOT currently held" / "never
# confirm" / "not mentioned anywhere" -- flat denials, not claims scoped to
# a specific product. Source-text presence can't be trusted for these (see
# comments above), so they're always treated as unsupported regardless of
# what check_restricted_claims finds in source_text. Everything else in
# RESTRICTED_TERM_CATEGORIES stays normally evidence-checked.
ALWAYS_UNSUPPORTED_TERMS: set[str] = {
    "atex", "iecex", "sil", "sil-2", "sil2",
    "life-safety", "life safety", "explosion proof", "explosion-proof",
    "ce mark", "ce marking",
    "helium", "methane", "sf6", "hydrocarbon", "refrigerant",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_term(text: str, term: str) -> bool:
    pattern = r"\b" + re.escape(term) + r"(?:es|s)?\b"
    return re.search(pattern, text) is not None


def _classify_category(text: str) -> tuple[str, list[str]]:
    """Returns (category, matched_terms) for the first category (in
    RESTRICTED_TERM_CATEGORIES order) with any term present in `text`, or
    (NONE_CATEGORY, []) if nothing restricted was found."""
    for category, terms in RESTRICTED_TERM_CATEGORIES.items():
        matched = [t for t in terms if _contains_term(text, t)]
        if matched:
            return category, matched
    return NONE_CATEGORY, []


@dataclass
class ClaimCheckResult:
    """category is one of RESTRICTED_TERM_CATEGORIES' keys, or "None" if
    nothing restricted was found in the question or draft answer.

    is_blocked is True when a restricted term was matched AND that term
    does not itself appear anywhere in the retrieved source text (spec
    12.3: "If the claim is unsupported, block external wording and
    request human review."). This checks the actual evidence text, not
    retrieval similarity -- a claim can be well-supported even at Low
    confidence (e.g. the exact restricted term is present in a chunk that
    just didn't rank as similar as it could have), and conversely a
    confidently-retrieved chunk doesn't make an answer that goes beyond
    what that chunk actually says any less unsupported.

    unsupported_terms lists which matched terms specifically weren't found
    in the source text -- this is why a result is blocked, for auditability.
    """

    category: str
    is_blocked: bool
    matched_terms: list[str]
    unsupported_terms: list[str]


def check_restricted_claims(question: str, answer_text: str, source_text: str) -> ClaimCheckResult:
    """Scans both the question and the draft answer (spec 12.3) for
    restricted-claim terms. The question's category takes priority when
    both are present (it's the more reliable signal of what's actually
    being asked); matched_terms is the union of both, for auditability.

    source_text is the retrieved evidence actually cited as sources for
    this answer (e.g. the concatenated text of retriever.retrieve()'s
    matches) -- a term is "supported" only if it appears there too, unless
    it's in ALWAYS_UNSUPPORTED_TERMS (flat denials like ATEX/CE/the MEMS
    gas list, where source-text presence doesn't mean confirmation -- see
    that set's docstring). Pass an empty string when there were no sources
    at all; every matched term is then correctly unsupported.
    """
    question_norm = _normalize(question)
    answer_norm = _normalize(answer_text)
    source_norm = _normalize(source_text)

    q_category, q_matches = _classify_category(question_norm)
    a_category, a_matches = _classify_category(answer_norm)

    category = q_category if q_category != NONE_CATEGORY else a_category
    matched_terms = sorted(set(q_matches) | set(a_matches))
    unsupported_terms = [
        t for t in matched_terms
        if t in ALWAYS_UNSUPPORTED_TERMS or not _contains_term(source_norm, t)
    ]

    is_blocked = category != NONE_CATEGORY and bool(unsupported_terms)

    return ClaimCheckResult(
        category=category,
        is_blocked=is_blocked,
        matched_terms=matched_terms,
        unsupported_terms=unsupported_terms,
    )


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    example_cases = [
        (
            "Is this ATEX certified?",
            "The product is certified for hazardous area use according to IS/IEC "
            "standards and is PESO approved for Zone 1 Gas Group IIC, but there is "
            "no explicit mention of ATEX certification.",
            # Real retrieved evidence never mentions ATEX (it's "in process",
            # per data/restricted_claims.md) -- "atex" stays unsupported.
            "FIXaHY is PESO approved for Gas Group IIC Zone 1. Tested at a "
            "3rd party BASEEFA accredited lab per IS/IEC 60079-11 and 60079-0.",
            "Certification", True,
        ),
        (
            "Is the FIXaHY sensor PESO approved for hazardous areas?",
            "Yes, the FIXaHY sensor is PESO approved for hazardous areas, "
            "specifically for Gas Group IIC Zone 1.",
            # Same evidence, but here the matched terms (peso, approved,
            # hazardous area, gas group, zone 1) all appear in it verbatim.
            "FIXaHY-G/P/E-4220MA-RRNNVVII - PESO Approved for Gas Group IIC "
            "Zone 1. The product is approved by PESO for deployment in "
            "hazardous area Zone 1.",
            "Certification", False,
        ),
        (
            "What is the price per unit?",
            "I do not have approved information for this. Please check with the "
            "technical or management team.",
            "",  # no sources at all
            "Pricing", True,
        ),
        (
            "What is the warranty period for Multi Nano Sense products?",
            "The warranty period is 12 months from the date of invoice.",
            "Multi Nano Sense provides a warranty against defective parts for "
            "12 months from the date of invoice.",
            "Pricing", False,
        ),
        (
            # Per data/restricted_claims.md: this is the MEMS platform's
            # roadmap target, not any current product's spec -- must be
            # blocked even though the roadmap slide's own text (used here
            # as source_text) genuinely contains all these gas names.
            "What gases can the MEMS sensor detect besides hydrogen?",
            "Besides hydrogen, the MEMS sensor can detect helium, methane, SF6, "
            "hydrocarbons, and refrigerants.",
            "Leak Detection: Hydrogen, Helium, Methane, SF6, Hydrocarbons, "
            "Refrigerants and other Gases.",
            "Accuracy", True,
        ),
        (
            # Source-text presence alone would wrongly call this
            # "supported": the real approved deck's own line literally
            # contains the word "atex" while denying the claim.
            "Is this ATEX certified?",
            "Yes, the product is ATEX certified.",
            "Designed for regulated industrial and safety applications; "
            "Certifications: PESO (IS/IEC 60079-11 and IS/IEC 60079-0), "
            "ARAI (EMI/EMC), IP 68 (IS/IEC 60529). IECEx & ATEX in Process.",
            "Certification", True,
        ),
        (
            "Is the enclosure CE marked?",
            "Yes, the enclosure is CE marked.",
            "Enclosure: IP66 Pressure Die Cast 4-Port Flameproof Certified "
            "LM6 Aluminium Enclosure.",
            "Certification", True,
        ),
        (
            # The case this whole change was about: real evidence, Low
            # similarity score -- must NOT be blocked now.
            "Can AURIGA be deployed in a hazardous area?",
            "No, the AURIGA Portable Hydrogen Leak Detector should not be "
            "deployed in hazardous areas.",
            "Area of Deployment Safe Area. This instrument should not be "
            "deployed in a Hazardous Area.",
            "Certification", False,
        ),
        (
            # A confidence-based check would have missed this: good retrieval,
            # but the answer claims something the source never states.
            "Is the FIXaHY sensor ATEX certified?",
            "Yes, the FIXaHY sensor is ATEX certified for hazardous areas.",
            "FIXaHY-G/P/E-4220MA-RRNNVVII - PESO Approved for Gas Group IIC "
            "Zone 1. Ex ia IIC T6 Ga.",
            "Certification", True,
        ),
    ]

    for question, answer, source_text, expected_category, expected_blocked in example_cases:
        result = check_restricted_claims(question, answer, source_text)
        marker = "OK" if (result.category == expected_category and result.is_blocked == expected_blocked) else "MISMATCH"
        print(
            f"[{marker}] category={result.category:<14} blocked={result.is_blocked!s:<5} "
            f"unsupported={result.unsupported_terms} | {question}"
        )
