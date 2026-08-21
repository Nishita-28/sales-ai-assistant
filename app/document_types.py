"""Explicit document classification for data/approved_docs -- what an
admin assigns at upload time in the Documents tab. An explicit, stored
choice rather than a filename-pattern guess, since a heuristic like
"eligible as a product unless the name contains comparison/competitor/
guide" is too easy to fool by a document whose name just doesn't happen
to match.

Drives two things: which documents populate the Product Registry /
single-product scoping (Product Catalogue only), and which documents the
main Assistant can retrieve at all (see MAIN_ASSISTANT_EXCLUDED_TYPES).

Backed by Postgres (Neon) when app.db.is_postgres_enabled().
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from app import db

TYPES_PATH = Path("data/document_types.json")

# get_document_type() is called once per document in a loop, which
# against Postgres means one round trip per document -- a short TTL
# collapses those into one query per render. save_document_types() also
# clears the cache directly, so an admin's own edit never waits on the TTL.
_CACHE_TTL_SECONDS = 3
_cache: tuple[float, dict[str, str]] | None = None

PRODUCT_CATALOGUE = "Product Catalogue"
USE_CASE_GUIDE = "Use Case Guide"
TECHNICAL_GUIDE = "Technical Guide"
HISTORICAL_SALES_RECORD = "Historical Sales Record"
INTERNAL_SALES_STRATEGY = "Competitive & Internal Strategy"
SALES_METHODOLOGY_REFERENCE = "Golden Frameworks & Sales Tactics"
OTHER = "Other"

ALL_TYPES = [
    PRODUCT_CATALOGUE,
    USE_CASE_GUIDE,
    TECHNICAL_GUIDE,
    HISTORICAL_SALES_RECORD,
    INTERNAL_SALES_STRATEGY,
    SALES_METHODOLOGY_REFERENCE,
    OTHER,
]

# Types excluded from the main Assistant -- a retrieved chunk there goes
# straight into a rep-visible answer with no guardrail in between, unlike
# Discovery Questions (stays internal) or Sales Aid (Claim-Checker gated).
# Internal Sales Strategy and Sales Methodology Reference are both
# internal coaching content, never meant to reach a rep-facing answer
# that could get relayed to a customer. OTHER is deliberately not
# included -- an uncategorized document isn't necessarily unsafe, so it
# behaves like Use Case Guide/Technical Guide (open, not Registry-eligible).
MAIN_ASSISTANT_EXCLUDED_TYPES = {INTERNAL_SALES_STRATEGY, SALES_METHODOLOGY_REFERENCE}


def load_document_types() -> dict[str, str]:
    """Returns {} if nothing is saved yet, rather than raising."""
    global _cache
    if db.is_postgres_enabled():
        now = time.time()
        if _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
            return _cache[1]
        db.ensure_schema()
        rows = db.fetch_all("SELECT document_name, doc_type FROM document_types")
        result = {row["document_name"]: row["doc_type"] for row in rows}
        _cache = (now, result)
        return result

    if not TYPES_PATH.exists():
        return {}
    return json.loads(TYPES_PATH.read_text(encoding="utf-8"))


def save_document_types(types: dict[str, str]) -> None:
    global _cache
    if db.is_postgres_enabled():
        db.ensure_schema()
        with db.get_connection() as conn:
            conn.execute("DELETE FROM document_types")
            for name, doc_type in types.items():
                conn.execute(
                    "INSERT INTO document_types (document_name, doc_type) VALUES (%s, %s)",
                    (name, doc_type),
                )
        _cache = None
        return

    TYPES_PATH.parent.mkdir(parents=True, exist_ok=True)
    TYPES_PATH.write_text(json.dumps(types, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def get_document_type(document_name: str) -> Optional[str]:
    """The assigned type, or None if this document was never classified
    (e.g. added via a CLI path that bypassed the Admin upload form)."""
    return load_document_types().get(document_name)


def set_document_type(document_name: str, doc_type: str) -> None:
    if doc_type not in ALL_TYPES:
        raise ValueError(f"Unknown document type: {doc_type!r}")
    types = load_document_types()
    types[document_name] = doc_type
    save_document_types(types)


def remove_document_type(document_name: str) -> None:
    types = load_document_types()
    if document_name in types:
        del types[document_name]
        save_document_types(types)


def is_product_catalogue(document_name: str) -> bool:
    """True only if explicitly classified as Product Catalogue --
    deliberately NOT the default for an unclassified document. An
    unclassified document stays invisible to the registry until someone
    consciously classifies it, rather than being silently treated as a
    purchasable product."""
    return get_document_type(document_name) == PRODUCT_CATALOGUE


def is_excluded_from_main_assistant(document_name: str) -> bool:
    doc_type = get_document_type(document_name)
    return doc_type is not None and doc_type in MAIN_ASSISTANT_EXCLUDED_TYPES
