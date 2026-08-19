"""Stores customer requirement submissions in a local SQLite database, so
a sales rep's technical notes from a call become a clean record for
production instead of a scattered email or notebook entry.

Backed by Postgres (Neon) when app.db.is_postgres_enabled() -- see
app.deals_store's module docstring for why (Streamlit Community Cloud's
ephemeral filesystem) and why returning plain dicts instead of
sqlite3.Row is safe here (every caller uses key-based access only).
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app import db

DB_PATH = Path("data/requirements.db")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requirements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            customer_name TEXT NOT NULL,
            company TEXT NOT NULL,
            application TEXT NOT NULL DEFAULT '',
            product_family TEXT NOT NULL DEFAULT '',
            install_type TEXT NOT NULL DEFAULT '',
            num_detectors INTEGER NOT NULL DEFAULT 1,
            comm_protocols TEXT NOT NULL DEFAULT '',
            certifications TEXT NOT NULL DEFAULT '',
            installation_area TEXT NOT NULL DEFAULT '',
            hazard_zone TEXT NOT NULL DEFAULT '',
            temp_min REAL,
            temp_max REAL,
            target_gas TEXT NOT NULL DEFAULT '',
            sensing_range TEXT NOT NULL DEFAULT '',
            accuracy TEXT NOT NULL DEFAULT '',
            environmental TEXT NOT NULL DEFAULT '',
            probe_length TEXT NOT NULL DEFAULT '',
            suggested_code TEXT NOT NULL DEFAULT '',
            additional_requirements TEXT NOT NULL DEFAULT '',
            extra_fields TEXT NOT NULL DEFAULT '',
            deal_id INTEGER
        )
        """
    )
    # Migrates databases created before these columns existed.
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(requirements)")}
    for column in ("product_family", "suggested_code", "extra_fields"):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE requirements ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
    if "deal_id" not in existing_columns:
        # Nullable, no default -- unlike the TEXT columns above, this is a
        # real reference to deals.db (see app.deals_store), not free text.
        # Null for any requirement submitted before deal-linkage existed,
        # or for one submitted without an active deal.
        conn.execute("ALTER TABLE requirements ADD COLUMN deal_id INTEGER")
    return conn


def record_requirement(
    customer_name: str,
    company: str,
    application: str = "",
    product_family: str = "",
    install_type: str = "",
    num_detectors: int = 1,
    comm_protocols: str = "",
    certifications: str = "",
    installation_area: str = "",
    hazard_zone: str = "",
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
    target_gas: str = "",
    sensing_range: str = "",
    accuracy: str = "",
    environmental: str = "",
    probe_length: str = "",
    suggested_code: str = "",
    additional_requirements: str = "",
    extra_fields: str = "",
    deal_id: Optional[int] = None,
) -> int:
    """Stores one customer requirement submission. Returns the new row id.
    extra_fields is a JSON object string ({field_key: answer}) for whatever
    admin-added custom fields (see app.requirements_fields) existed on the
    form at submission time -- kept generic here since new custom fields
    can appear at any time without a schema change. deal_id links this
    submission to a deals.db record (see app.deals_store) -- None for a
    requirement captured with no active deal."""
    now = datetime.now().isoformat(timespec="seconds")
    values = (
        now, customer_name, company,
        application, product_family, install_type, num_detectors, comm_protocols,
        certifications, installation_area, hazard_zone, temp_min, temp_max, target_gas,
        sensing_range, accuracy, environmental, probe_length, suggested_code,
        additional_requirements, extra_fields, deal_id,
    )
    columns_sql = """
                created_at, customer_name, company, application, product_family,
                install_type, num_detectors, comm_protocols, certifications,
                installation_area, hazard_zone, temp_min, temp_max, target_gas,
                sensing_range, accuracy, environmental, probe_length, suggested_code,
                additional_requirements, extra_fields, deal_id
    """
    if db.is_postgres_enabled():
        db.ensure_schema()
        return db.execute_returning(
            f"INSERT INTO requirements ({columns_sql}) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id",
            values,
        )
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            f"INSERT INTO requirements ({columns_sql}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        return cursor.lastrowid


def list_requirements(limit: int = 200) -> list[Any]:
    """Most recent requirement submissions, newest first."""
    if db.is_postgres_enabled():
        db.ensure_schema()
        return db.fetch_all("SELECT * FROM requirements ORDER BY created_at DESC LIMIT %s", (limit,))
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM requirements ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()


def list_requirements_for_deal(deal_id: int) -> list[Any]:
    """Every requirement submission linked to a specific deal, oldest
    first. The read side of the deal_id column recorded by
    record_requirement() -- without this, deal_id is written but never
    consulted anywhere, and a deal can't actually show what's been
    captured for it."""
    if db.is_postgres_enabled():
        db.ensure_schema()
        return db.fetch_all(
            "SELECT * FROM requirements WHERE deal_id = %s ORDER BY created_at ASC", (deal_id,)
        )
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM requirements WHERE deal_id = ? ORDER BY created_at ASC", (deal_id,)
        ).fetchall()


def delete_requirement(requirement_id: int) -> None:
    """Permanently removes a requirement submission from the database."""
    if db.is_postgres_enabled():
        db.execute("DELETE FROM requirements WHERE id = %s", (requirement_id,))
        return
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM requirements WHERE id = ?", (requirement_id,))
