"""Stores Deal records in a local SQLite database -- the persistence
layer behind Discovery's qualification-question flow (see
app/discovery_generator.py), so a rep's captured MEDDPICC answers for a
customer survive between visits instead of living only in browser
session state.

Deliberately minimal, per the agreed V1 scope: a deal is just an id,
customer_name, company, use_case, the 8 MEDDPICC fields, and its last
Right-to-Win generation, qualification-question generation, and product
recommendation (see save_discovery/save_qualification/
save_recommendation and their get_ counterparts). No stage,
stakeholders, interaction history, framework selection, or risk score
yet -- those depend on real usage data existing first, which this
table is what starts accumulating.

Keyed by an internal id, not by customer_name/company -- the same
company can have multiple distinct deals (e.g. "ABC Industries --
Electrolyzer Project" and "ABC Industries -- Portable Detector Order"),
which a customer/company key would silently collapse into one record.
customer_name/company are for display and search, not identity.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path("data/deals.db")

# The 8 MEDDPICC fields, in the order they appear in "Sales Methodology
# and Qualification Frameworks Reference" -- (column_name, label,
# default_question). Single source of truth: both deals_store (schema)
# and discovery_generator (which fields are still blank, and what to ask
# about them) import this rather than each keeping their own copy.
MEDDPICC_FIELDS: list[tuple[str, str, str]] = [
    ("metrics", "Metrics", "What result are you trying to achieve, and how will you measure whether this solves it?"),
    ("economic_buyer", "Economic Buyer", "Who ultimately signs off on a purchase like this?"),
    ("decision_criteria", "Decision Criteria", "What criteria will you use to decide between us and any alternative?"),
    ("decision_process", "Decision Process", "What does the approval process look like from here to a purchase order?"),
    ("paper_process", "Paper Process", "Once you decide, what's the internal paperwork/procurement process and how long does it typically take?"),
    ("identify_pain", "Identify Pain", "What's driving this now, as opposed to six months ago or six months from now?"),
    ("champion", "Champion", "Who on your side is most invested in getting this solved?"),
    ("competition", "Competition", "Who else are you looking at for this?"),
]
MEDDPICC_FIELD_KEYS = [key for key, _, _ in MEDDPICC_FIELDS]


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    field_columns = ",\n            ".join(f"{key} TEXT NOT NULL DEFAULT ''" for key in MEDDPICC_FIELD_KEYS)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS deals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            company TEXT NOT NULL,
            use_case TEXT NOT NULL DEFAULT '',
            {field_columns},
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_recommendation TEXT,
            last_discovery TEXT,
            last_qualification TEXT
        )
        """
    )
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(deals)")}
    # Nullable JSON blobs, all added after MEDDPICC_FIELDS existed --
    # last_recommendation (discovery_generator.RecommendationResult.
    # to_dict()), last_discovery ({"right_to_win": [RightToWinPoint.
    # to_dict(), ...], "answers": {question: answer}}), and
    # last_qualification ([QualificationQuestion.to_dict(), ...]).
    # Proven necessary by direct feedback: Right-to-Win results and a
    # rep's typed answers used to live only in browser session state,
    # so leaving the page (or even just navigating to a different page
    # and back) silently lost everything, forcing a full regenerate.
    for column in ("last_recommendation", "last_discovery", "last_qualification"):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE deals ADD COLUMN {column} TEXT")
    return conn


def _save_json_column(deal_id: int, column: str, value: Any) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            f"UPDATE deals SET {column} = ?, updated_at = ? WHERE id = ?",
            (json.dumps(value, ensure_ascii=False), datetime.now().isoformat(timespec="seconds"), deal_id),
        )


def _get_json_column(deal: sqlite3.Row, column: str) -> Optional[Any]:
    raw = deal[column]
    return json.loads(raw) if raw else None


def create_deal(customer_name: str, company: str, use_case: str = "") -> int:
    """Creates a new deal and returns its id. All 8 MEDDPICC fields start
    blank -- filled in over time via update_deal_fields()."""
    now = datetime.now().isoformat(timespec="seconds")
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO deals (customer_name, company, use_case, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (customer_name, company, use_case, now, now),
        )
        return cursor.lastrowid


def get_deal(deal_id: int) -> Optional[sqlite3.Row]:
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM deals WHERE id = ?", (deal_id,)).fetchone()


def list_deals(limit: int = 200) -> list[sqlite3.Row]:
    """Most recently updated deals first -- for a "which deal" picker
    (search/display by customer_name/company), never for identity."""
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM deals ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()


def update_deal_fields(deal_id: int, **fields: str) -> None:
    """Updates one or more of the 8 MEDDPICC fields (or use_case) for an
    existing deal. Only ever overwrites a field with a non-empty value --
    callers should not pass blanks for fields they don't intend to
    change, since this isn't a full-record replace."""
    allowed = set(MEDDPICC_FIELD_KEYS) | {"use_case"}
    updates = {k: v for k, v in fields.items() if k in allowed and v}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = [*updates.values(), datetime.now().isoformat(timespec="seconds"), deal_id]
    with closing(_connect()) as conn, conn:
        conn.execute(f"UPDATE deals SET {set_clause}, updated_at = ? WHERE id = ?", values)


def save_recommendation(deal_id: int, recommendation: dict[str, Any]) -> None:
    """Persists the JSON-serializable dict form of a RecommendationResult
    (see discovery_generator.RecommendationResult.to_dict()) as this
    deal's last recommendation -- so reopening a deal shows what was last
    recommended instead of losing it once the browser session ends."""
    _save_json_column(deal_id, "last_recommendation", recommendation)


def get_recommendation(deal: sqlite3.Row) -> Optional[dict[str, Any]]:
    """The deal's last saved recommendation dict, or None if it never got
    one. Kept as a plain dict here, not discovery_generator.
    RecommendationResult -- deals_store.py has no reason to import that
    module's types; callers (discovery_page.py) reconstruct it."""
    return _get_json_column(deal, "last_recommendation")


def save_discovery(
    deal_id: int,
    right_to_win: list[dict[str, Any]],
    answers: dict[str, str],
    sources: list[tuple[str, str]] = (),
) -> None:
    """Persists this deal's last Right-to-Win generation (a list of
    discovery_generator.RightToWinPoint.to_dict()) plus whatever answers
    the rep has typed against those questions so far, and the KB sources
    it was grounded in -- all three used to be session-only and vanished
    the moment the rep left the page."""
    _save_json_column(
        deal_id, "last_discovery", {"right_to_win": right_to_win, "answers": answers, "sources": list(sources)}
    )


def get_discovery(deal: sqlite3.Row) -> Optional[dict[str, Any]]:
    """{"right_to_win": [...], "answers": {...}}, or None if this deal
    has no saved Right-to-Win generation yet."""
    return _get_json_column(deal, "last_discovery")


def save_qualification(deal_id: int, qualification: list[dict[str, Any]]) -> None:
    """Persists this deal's last qualification-question generation (a
    list of discovery_generator.QualificationQuestion.to_dict()). The
    rep's actual answers live directly on the deal's own MEDDPICC
    columns (see update_deal_fields) -- this is only the LLM-phrased
    question/rationale text, so it doesn't have to be regenerated every
    time the deal is reopened."""
    _save_json_column(deal_id, "last_qualification", qualification)


def get_qualification(deal: sqlite3.Row) -> Optional[list[dict[str, Any]]]:
    return _get_json_column(deal, "last_qualification")


def missing_fields(deal: sqlite3.Row) -> list[tuple[str, str, str]]:
    """The subset of MEDDPICC_FIELDS still blank for this deal, in
    canonical order -- what Discovery should still ask about."""
    return [(key, label, question) for key, label, question in MEDDPICC_FIELDS if not deal[key].strip()]


def delete_deal(deal_id: int) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM deals WHERE id = ?", (deal_id,))
