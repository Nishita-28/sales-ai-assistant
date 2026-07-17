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
        "certification", "certified", "approval", "approved",
        
        "peso", "arai", "iec", "iecex", "atex", "sil",
        "emi", "emc",
        "hazardous", "explosion", "explosion proof", "intrinsically safe",
        "flameproof", "zone 1", "gas group", "ex ia",
        "accuracy", "legal", "regulatory", "life-safety",
    ],
    "pricing": [
        "price", "pricing", "quotation", "quote", "cost", "warranty",
        "delivery", "lead time", "amc",
    ],
    "customer_draft": [
        "draft", "email", "whatsapp", "customer reply", "customer response", "rewrite",
    ],
    "document_request": [
        "brochure", "document", "datasheet", "presentation", "pdf", "send", "file",
        "catalogue", "catalog",
    ],
    "application": [
        "application", "industrial", "monitoring", "installation", "install",
        "deploy", "deployment", "in-line analysis",
        
        "oil and gas", "oil & gas", "power plant", "manufacturing",
        "battery room", "hydrogen mobility", "fcev", "research lab",
        "testing laboratory", "testing & certification",
        "steel manufacturing", "process industries",
        "space industry", "aerospace",
        "portable leak detection",
     
    ],
    "product": [
        "product", "sensor", "detector", "feature", "specification",
        "interface", "hydrogen",
        "model", "device", "range", "leak detector",
        
        "rs485", "rs-485", "rs 485",
        "4-20 ma", "4-20ma", "4 20 ma",
     
        "mems", "electrochemical", "solid state",
        "helium", "methane", "sf6", "hydrocarbons", "refrigerants",
        "leak detection", "analyzer",
      
        "sensitivity", "response time", "ip rating", "ingress protection",
        
        "ip66", "ip 66", "ip67", "ip 67", "ip68", "ip 68",
        "output signal", "power supply",
        "nomenclature", "part number",
    ],
}

UNKNOWN_INTENT: Intent = "unknown"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_keyword(text: str, keyword: str) -> bool:
    """Word-boundary match, tolerant of a trailing "s"/"es" on the keyword
    (e.g. "certification" also matches "certifications", "detector" also
    matches "detectors") so plural phrasing doesn't need every keyword
    hand-listed twice, and so a keyword doesn't need a separate redundant
    entry just for its plural form. Doesn't handle irregular plurals (e.g.
    "accuracy" -> "accuracies") or verb tense (e.g. "deploy" -> "deployed")
    -- none of the current keywords need that."""
    pattern = r"\b" + re.escape(keyword) + r"(?:es|s)?\b"
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
        
        ("Is this suitable for battery room or oil & gas monitoring?", "application"),
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
        ("Is the ATEX-certified detector available?", "compliance"),
        # Doc-grounded coverage.
        ("Is the FIXaHY sensor PESO approved for hazardous areas?", "compliance"),
        ("Can this MEMS sensor also detect methane or SF6, not just hydrogen?", "product"),
        ("What's the response time and sensitivity of the portable detector?", "product"),
        ("What's the part number for the RS485 output variant?", "product"),
        ("Can this be deployed for battery room monitoring in a power plant?", "application"),
        ("Is this suitable for hydrogen mobility and FCEV bus applications?", "application"),
        ("What's the AMC and lead time after installation?", "pricing"),
        ("Can you send me the product catalogue?", "document_request"),
        ("Do we have certifications for this detector?", "compliance"),
        
        ("Is the enclosure IP68 rated?", "product"),
        ("Is this rated for hazardous zones?", "compliance"),
        
        # manufacturing"/"space industry" keywords no longer misfire here.
        ("Is the sensor probe small enough to fit in a tight space?", "product"),
        ("Do you have any space industry or aerospace customers?", "application"),
        ("Is this suitable for steel manufacturing or process industries?", "application"),
    ]

    for question, expected in example_questions:
        result = classify_intent(question)
        marker = "OK" if result == expected else "MISMATCH"
        print(f"[{marker}] intent={result:<16} expected={expected:<16} | {question}")
