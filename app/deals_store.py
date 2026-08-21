"""Stores Deal records in a local SQLite database -- the persistence
layer behind Discovery's qualification-question flow, so a rep's
captured MEDDPICC answers for a customer survive between visits instead
of living only in browser session state.

A deal is just an id, customer_name, company, use_case, the 8 MEDDPICC
fields, and its last Right-to-Win/qualification-question/recommendation
generation (see save_discovery/save_qualification/save_recommendation
and their get_ counterparts).

Keyed by an internal id, not customer_name/company -- the same company
can have multiple distinct deals, which a customer/company key would
silently collapse into one record.

Backed by Postgres (Neon) when app.db.is_postgres_enabled() -- otherwise
a deal captured on Streamlit Cloud's ephemeral filesystem would vanish on
the next redeploy. Falls back to local SQLite when DATABASE_URL isn't
set. Every function returns something dict-like regardless of backend --
every caller here uses key-based access (row["col"]), never positional,
so the sqlite3.Row/psycopg dict_row swap is safe."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app import db

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
    # Nullable JSON blobs, added after MEDDPICC_FIELDS existed -- persist
    # Right-to-Win results and a rep's typed answers across page
    # navigation, not just browser session state.
    for column in ("last_recommendation", "last_discovery", "last_qualification"):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE deals ADD COLUMN {column} TEXT")
    return conn


def _save_json_column(deal_id: int, column: str, value: Any) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    payload = json.dumps(value, ensure_ascii=False)
    if db.is_postgres_enabled():
        db.ensure_schema()
        db.execute(f"UPDATE deals SET {column} = %s, updated_at = %s WHERE id = %s", (payload, now, deal_id))
        return
    with closing(_connect()) as conn, conn:
        conn.execute(f"UPDATE deals SET {column} = ?, updated_at = ? WHERE id = ?", (payload, now, deal_id))


def _get_json_column(deal: Any, column: str) -> Optional[Any]:
    raw = deal[column]
    return json.loads(raw) if raw else None


def create_deal(customer_name: str, company: str, use_case: str = "") -> int:
    """Creates a new deal and returns its id. All 8 MEDDPICC fields start
    blank -- filled in over time via update_deal_fields()."""
    now = datetime.now().isoformat(timespec="seconds")
    if db.is_postgres_enabled():
        db.ensure_schema()
        return db.execute_returning(
            "INSERT INTO deals (customer_name, company, use_case, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (customer_name, company, use_case, now, now),
        )
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO deals (customer_name, company, use_case, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (customer_name, company, use_case, now, now),
        )
        return cursor.lastrowid


def get_deal(deal_id: int) -> Optional[Any]:
    if db.is_postgres_enabled():
        db.ensure_schema()
        return db.fetch_one("SELECT * FROM deals WHERE id = %s", (deal_id,))
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM deals WHERE id = ?", (deal_id,)).fetchone()


def list_deals(limit: int = 200) -> list[Any]:
    """Most recently updated deals first -- for a "which deal" picker
    (search/display by customer_name/company), never for identity."""
    if db.is_postgres_enabled():
        db.ensure_schema()
        return db.fetch_all("SELECT * FROM deals ORDER BY updated_at DESC LIMIT %s", (limit,))
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
    now = datetime.now().isoformat(timespec="seconds")
    if db.is_postgres_enabled():
        db.ensure_schema()
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        values = (*updates.values(), now, deal_id)
        db.execute(f"UPDATE deals SET {set_clause}, updated_at = %s WHERE id = %s", values)
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = [*updates.values(), now, deal_id]
    with closing(_connect()) as conn, conn:
        conn.execute(f"UPDATE deals SET {set_clause}, updated_at = ? WHERE id = ?", values)


def save_recommendation(deal_id: int, recommendation: dict[str, Any]) -> None:
    """Persists the JSON-serializable dict form of a RecommendationResult
    (see discovery_generator.RecommendationResult.to_dict()) as this
    deal's last recommendation -- so reopening a deal shows what was last
    recommended instead of losing it once the browser session ends."""
    _save_json_column(deal_id, "last_recommendation", recommendation)


def get_recommendation(deal: Any) -> Optional[dict[str, Any]]:
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
    it was grounded in."""
    _save_json_column(
        deal_id, "last_discovery", {"right_to_win": right_to_win, "answers": answers, "sources": list(sources)}
    )


def get_discovery(deal: Any) -> Optional[dict[str, Any]]:
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


def get_qualification(deal: Any) -> Optional[list[dict[str, Any]]]:
    return _get_json_column(deal, "last_qualification")


def missing_fields(deal: Any) -> list[tuple[str, str, str]]:
    """The subset of MEDDPICC_FIELDS still blank for this deal, in
    canonical order -- what Discovery should still ask about."""
    return [(key, label, question) for key, label, question in MEDDPICC_FIELDS if not deal[key].strip()]


def delete_deal(deal_id: int) -> None:
    if db.is_postgres_enabled():
        db.execute("DELETE FROM deals WHERE id = %s", (deal_id,))
        return
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM deals WHERE id = ?", (deal_id,))
