#python -m app.claim_checker
"""Implements the "Claim checker" component (spec 4.1) and the claim-
checking rules in Implementation Details 12.3: scans a question and its
draft answer for restricted-claim terms, decides which risk category
applies, and blocks customer-facing wording when a matched term isn't
backed by the actual retrieved source text.

Deliberately keyword-based, not an LLM/ML classifier -- spec 12.3: "Keep
this simple initially using keyword rules; ML classification can be a
later improvement." Same reasoning as intent.py's classifier.

Risk categories match streamlit_app.py's RISK_COLORS keys exactly: None,
Certification, Accuracy, Safety, Pricing, Legal, Delivery.

The actual keyword/category/always-unsupported policy lives in
data/restricted_claims.yaml (see app/restricted_policy.py), not here --
it's edited through the Admin page, and _load_policy() below picks up
changes automatically (checked by file mtime, so a save takes effect on
the very next check without restarting the app, but repeated checks
against an unchanged file don't reparse it every time).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.restricted_policy import PolicyError, build_lookup, load_entries

NONE_CATEGORY = "None"

POLICY_PATH = Path("data/restricted_claims.yaml")

# (mtime the cache was built from, categories dict, always_unsupported set)
_policy_cache: Optional[tuple[float, dict[str, list[str]], set[str]]] = None


def _load_policy() -> tuple[dict[str, list[str]], set[str]]:
    """Categories are checked in the order they first appear in the policy
    file (first matching category wins) -- mirrors the old hardcoded
    dict's insertion-order behaviour, just sourced from disk now."""
    global _policy_cache

    try:
        mtime = POLICY_PATH.stat().st_mtime
    except OSError as e:
        raise PolicyError(f"Restricted-claims policy file not found at {POLICY_PATH}: {e}") from e

    if _policy_cache is not None and _policy_cache[0] == mtime:
        return _policy_cache[1], _policy_cache[2]

    categories, always_unsupported = build_lookup(load_entries(POLICY_PATH))
    _policy_cache = (mtime, categories, always_unsupported)
    return categories, always_unsupported


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_term(text: str, term: str) -> bool:
    pattern = r"\b" + re.escape(term) + r"(?:es|s)?\b"
    return re.search(pattern, text) is not None


# "approved"/"approval" are unambiguous when a user asks them ("Is this
# approved for X?"), but not when they show up in the LLM's OWN answer
# text: the system prompt requires it to talk about the "approved
# knowledge base"/"approved information" even when saying it found
# nothing, so an ordinary "the approved excerpts do not specify X" answer
# would otherwise falsely trigger Certification on every single
# unrelated, unsupported-looking answer. Excluded only from answer-text
# classification below, not from the question.
ANSWER_TEXT_EXCLUDED_TERMS = {"approved", "approval"}


def _classify_category(text: str, exclude: set[str] = frozenset()) -> tuple[str, list[str]]:
    """Returns (category, matched_terms) for the first category (in
    policy-file order, see _load_policy) with any term present in `text`
    (skipping any term in `exclude`), or (NONE_CATEGORY, []) if nothing
    restricted was found."""
    categories, _ = _load_policy()
    for category, terms in categories.items():
        matched = [t for t in terms if t not in exclude and _contains_term(text, t)]
        if matched:
            return category, matched
    return NONE_CATEGORY, []


@dataclass
class ClaimCheckResult:
    """category is one of the categories named in data/restricted_claims.yaml,
    or "None" if nothing restricted was found in the question or draft answer.

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
    ANSWER_TEXT_EXCLUDED_TERMS is skipped when classifying answer_text
    specifically (see its own docstring).

    source_text is the retrieved evidence actually cited as sources for
    this answer (e.g. the concatenated text of retriever.retrieve()'s
    matches) -- a term is "supported" only if it appears there too, unless
    its policy entry has always_unsupported: true (flat denials like
    ATEX/CE/the MEMS gas list, where source-text presence doesn't mean
    confirmation -- see data/restricted_claims.yaml). Pass an empty string
    when there were no sources at all; every matched term is then
    correctly unsupported.
    """
    question_norm = _normalize(question)
    answer_norm = _normalize(answer_text)
    source_norm = _normalize(source_text)

    _, always_unsupported = _load_policy()

    q_category, q_matches = _classify_category(question_norm)
    a_category, a_matches = _classify_category(answer_norm, exclude=ANSWER_TEXT_EXCLUDED_TERMS)

    category = q_category if q_category != NONE_CATEGORY else a_category
    matched_terms = sorted(set(q_matches) | set(a_matches))
    unsupported_terms = [
        t for t in matched_terms
        if t in always_unsupported or not _contains_term(source_norm, t)
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
            # per data/restricted_claims.yaml) -- "atex" stays unsupported.
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
            # Real reported bug: an ordinary "not found" answer's own
            # boilerplate ("the approved excerpts...") was misread as a
            # certification claim. Nothing restricted is actually being
            # asked or claimed here.
            "What is the weight of the VISION H2 LD device?",
            "The approved excerpts do not specify the weight of the VISION H2 LD device.",
            "Product Ordering Information: users can select the variant "
            "from the options provided in this datasheet.",
            "None", False,
        ),
        (
            # Per data/restricted_claims.yaml: this is the MEMS platform's
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
