"""Stores generated Sales Aids in a local SQLite database. Previously
this page had zero persistence at all -- a result lived only in browser
session state and was gone on refresh.

One row per generation, not a single "latest" slot -- unlike Discovery's
recommendation (a single evolving verdict, see deals_store.
save_recommendation), a rep can reasonably generate several distinct
Sales Aids for the same deal over time (different competitors, different
framing at different points in the sales cycle), and losing that history
would throw away real, reusable material. Mirrors requirements_store.py's
shape for the same reason requirements keeps full history too.

Includes the deal_id read side (list_sales_aids_for_deal) from the
start -- a prior review of the deal system found that deal_id had been
added to requirements.db but nothing ever queried it back out, so a
deal's linked data was invisible everywhere except by writing raw SQL.
Not repeating that here.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path("data/sales_aids.db")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sales_aids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            deal_id INTEGER,
            use_case TEXT NOT NULL DEFAULT '',
            compare_against TEXT NOT NULL DEFAULT '',
            result TEXT NOT NULL
        )
        """
    )
    return conn


def record_sales_aid(
    use_case: str, compare_against: str, result: dict[str, Any], deal_id: Optional[int] = None
) -> int:
    """Stores one generated Sales Aid. result is
    sales_aid_generator.SalesAidResult.to_dict(). Returns the new row
    id. deal_id links this generation to a deals.db record -- None for
    one generated with no active deal."""
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO sales_aids (created_at, deal_id, use_case, compare_against, result) VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(timespec="seconds"), deal_id, use_case, compare_against,
                json.dumps(result, ensure_ascii=False),
            ),
        )
        return cursor.lastrowid


def list_sales_aids(limit: int = 200) -> list[sqlite3.Row]:
    """Most recently generated Sales Aids, newest first."""
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM sales_aids ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()


def list_sales_aids_for_deal(deal_id: int) -> list[sqlite3.Row]:
    """Every Sales Aid generated for a specific deal, oldest first --
    the read side of the deal_id column recorded by record_sales_aid()."""
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM sales_aids WHERE deal_id = ? ORDER BY created_at ASC", (deal_id,)
        ).fetchall()


def get_result(row: sqlite3.Row) -> dict[str, Any]:
    """The stored result dict for one row -- callers (sales_aid_page.py)
    reconstruct sales_aid_generator.SalesAidResult from this."""
    return json.loads(row["result"])


def delete_sales_aid(sales_aid_id: int) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM sales_aids WHERE id = ?", (sales_aid_id,))
