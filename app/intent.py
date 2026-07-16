from __future__ import annotations

import re
from typing import Literal


Intent = Literal[
    "product",
    "application",
    "compliance",
    "pricing",
    "customer_draft",
    "document_request",
    "unknown",
]

INTENT_KEYWORDS: dict[Intent, list[str]] = {
    "compliance": [
        "certification", "certified", "atex", "iecex", "sil",
        "hazardous", "explosion", "intrinsically safe", "accuracy",
        "legal", "regulatory", "life-safety",
        # Deliberately NOT adding bare "safety" or "warranty" here even
        # though the spec mentions them under compliance: compliance is
        # checked first, so either one would shadow more specific matches
        # further down -- "warranty" is already pricing's job, and a bare
        # "safety" would swallow the "home safety" application phrase
        # below before it ever gets a chance to match.
    ],
    "pricing": [
        "price", "pricing", "quotation", "quote", "cost", "warranty", "delivery",
    ],
    "customer_draft": [
        "draft", "email", "whatsapp", "customer reply", "customer response", "rewrite",
    ],
    "document_request": [
        "brochure", "document", "datasheet", "presentation", "pdf", "send", "file",
    ],
    "application": [
    "application","industrial","green energy","monitoring","installation","install","deploy","deployment","oil and gas","process monitoring","home safety",
],
    "product": [
        "product", "sensor", "detector", "feature", "specification",
        "interface", "hydrogen",
        "model", "device", "range", "leak detector",
        # Variants of the same terms users actually type differently
        # (with/without a hyphen, with/without a space).
        "rs485", "rs-485", "rs 485",
        "4-20 ma", "4-20ma", "4 20 ma",
    ],
}

UNKNOWN_INTENT: Intent = "unknown"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_keyword(text: str, keyword: str) -> bool:
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, text) is not None


def classify_intent(question: str) -> Intent:
    normalized = _normalize(question)

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(_contains_keyword(normalized, keyword) for keyword in keywords):
            return intent

    return UNKNOWN_INTENT


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_questions: list[tuple[str, Intent]] = [
        ("Can we detect hydrogen with this sensor?", "product"),
        ("Is this suitable for industrial oil and gas monitoring?", "application"),
        ("Is the sensor ATEX certified for hazardous areas?", "compliance"),
        ("What's the pricing and delivery time for 50 units?", "pricing"),
        ("Can you draft a customer reply email about the leak detector?", "customer_draft"),
        ("Please send me the datasheet and brochure as a PDF.", "document_request"),
        ("What's the weather like today?", "unknown"),
        ("Does regulatory approval require life-safety testing?", "compliance"),
        ("Is this suitable for home safety monitoring?", "application"),
        ("What's the device model range and specification?", "product"),
        ("What does the RS-485 interface support?", "product"),
        ("How much does it cost?", "pricing"),
        ("Send me the product brochure.", "document_request"),
        ("Can you rewrite this email?", "customer_draft"),
        ("Is it certified?", "compliance"),
        ("How do I install the detector?", "application"),
        ("Hello", "unknown"),
        ("asdfghjkl", "unknown"),
        ("Can I get a quotation?", "pricing"),
        ("Please send the PDF.", "document_request"),
        ("Is the detector intrinsically safe?", "compliance"),
        ("Is the ATEX-certified detector available?", "compliance")
    ]

    for question, expected in example_questions:
        result = classify_intent(question)
        marker = "OK" if result == expected else "MISMATCH"
        print(f"[{marker}] intent={result:<16} expected={expected:<16} | {question}")