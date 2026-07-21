#python -m app.claim_checker
"""Flags restricted claims (certifications, pricing, safety, etc.) in a
question or answer and checks whether they're backed by the source text."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.restricted_policy import PolicyError, build_lookup, load_entries

NONE_CATEGORY = "None"

POLICY_PATH = Path("data/restricted_claims.yaml")

_policy_cache: Optional[tuple[float, dict[str, list[str]], set[str]]] = None


def _load_policy() -> tuple[dict[str, list[str]], set[str]]:
    """Loads and caches the restricted-claims policy."""
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


# Answers mention "approved" even when nothing was found, so exclude it
# here to avoid false Certification flags on ordinary not-found replies.
ANSWER_TEXT_EXCLUDED_TERMS = {"approved", "approval"}


def _classify_category(text: str, exclude: set[str] = frozenset()) -> tuple[str, list[str]]:
    """Returns the first matching category and its matched terms, or
    (NONE_CATEGORY, []) if nothing matches."""
    categories, _ = _load_policy()
    for category, terms in categories.items():
        matched = [t for t in terms if t not in exclude and _contains_term(text, t)]
        if matched:
            return category, matched
    return NONE_CATEGORY, []


@dataclass
class ClaimCheckResult:
    """Result of a claim check: the risk category, whether it's blocked,
    and which terms triggered that."""

    category: str
    is_blocked: bool
    matched_terms: list[str]
    unsupported_terms: list[str]


def check_restricted_claims(question: str, answer_text: str, source_text: str) -> ClaimCheckResult:
    """Checks a question and its draft answer for restricted claims, and
    determines whether each one is backed by source_text. Pass an empty
    source_text if there were no sources -- every match is then unsupported."""
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
            # ATEX certification is still pending, so this stays unsupported.
            "FIXaHY is PESO approved for Gas Group IIC Zone 1. Tested at a "
            "3rd party BASEEFA accredited lab per IS/IEC 60079-11 and 60079-0.",
            "Certification", True,
        ),
        (
            "Is the FIXaHY sensor PESO approved for hazardous areas?",
            "Yes, the FIXaHY sensor is PESO approved for hazardous areas, "
            "specifically for Gas Group IIC Zone 1.",
            # Same evidence, but this time it's stated explicitly.
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
            # A generic "not found" reply shouldn't be flagged as a claim.
            "What is the weight of the VISION H2 LD device?",
            "The approved excerpts do not specify the weight of the VISION H2 LD device.",
            "Product Ordering Information: users can select the variant "
            "from the options provided in this datasheet.",
            "None", False,
        ),
        (
            # A future roadmap feature, not a current product -- should
            # block even though the source text mentions these gases.
            "What gases can the MEMS sensor detect besides hydrogen?",
            "Besides hydrogen, the MEMS sensor can detect helium, methane, SF6, "
            "hydrocarbons, and refrigerants.",
            "Leak Detection: Hydrogen, Helium, Methane, SF6, Hydrocarbons, "
            "Refrigerants and other Gases.",
            "Accuracy", True,
        ),
        (
            # The word "atex" appears in the source, but only to deny it.
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
            # Weak match, but the claim is genuinely supported.
            "Can AURIGA be deployed in a hazardous area?",
            "No, the AURIGA Portable Hydrogen Leak Detector should not be "
            "deployed in hazardous areas.",
            "Area of Deployment Safe Area. This instrument should not be "
            "deployed in a Hazardous Area.",
            "Certification", False,
        ),
        (
            # Good retrieval, but the answer claims more than the source says.
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
