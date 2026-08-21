"""Postgres (Neon) connection layer -- durable persistence for every
admin-mutable store when DATABASE_URL is set. Streamlit Cloud's
filesystem is ephemeral, so stores fall back to local SQLite/JSON/YAML
when it isn't (no live DB needed locally or in tests). Use Neon's
*pooled* connection string, not the direct one -- a Streamlit app holds
several concurrent connections across sessions/reruns.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class DatabaseUnavailableError(RuntimeError):
    """Raised when DATABASE_URL is set but a connection or query fails.
    Never falls back to local files silently -- stores surface this as a
    real st.error() instead, so an admin's change isn't lost unnoticed."""


def get_database_url() -> Optional[str]:
    """None means "use local file storage". Checks st.secrets first, then
    the environment; wrapped in try/except since st.secrets raises when
    no secrets.toml exists (the normal local-dev case)."""
    try:
        import streamlit as st

        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
    except Exception:
        pass
    return os.environ.get("DATABASE_URL")


def is_postgres_enabled() -> bool:
    return bool(get_database_url())


_pool: Optional[ConnectionPool] = None


def _get_pool() -> ConnectionPool:
    """Process-level pool of warm connections, opened lazily -- avoids a
    fresh handshake to Neon on every query. check_connection replaces a
    dead connection before handing it out, since Neon's free tier
    suspends after 5 min idle; max_idle=240 stays under that window."""
    global _pool
    if _pool is None:
        url = get_database_url()
        if not url:
            raise DatabaseUnavailableError("DATABASE_URL is not configured.")
        try:
            _pool = ConnectionPool(
                url,
                min_size=1,
                max_size=5,
                kwargs={"row_factory": dict_row, "connect_timeout": 10},
                open=True,
                timeout=10,
                check=ConnectionPool.check_connection,
                max_idle=240,
            )
        except Exception as e:
            raise DatabaseUnavailableError(f"Could not connect to the Postgres database: {e}") from e
    return _pool


# Bounded, not infinite -- this same path backs the shared cold-start sync
# in streamlit_app.py, and an unbounded retry there would hang the app for
# every visitor waiting on it together.
_CONNECT_RETRY_ATTEMPTS = 3
_CONNECT_RETRY_DELAY_SECONDS = 3


def _discard_pool() -> None:
    """Drops the pool so the next _get_pool() call builds a fresh one --
    a failed attempt may mean the pool itself is unhealthy, not just one
    bad connection."""
    global _pool, _schema_ensured
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass
        _pool = None
        _schema_ensured = False


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Borrows a connection from the pool; commits on clean exit, rolls
    back and raises DatabaseUnavailableError on failure. Retries a few
    times first, since waking a suspended Neon compute can be slower than
    one attempt allows."""
    last_error: Optional[Exception] = None
    for attempt in range(_CONNECT_RETRY_ATTEMPTS):
        if attempt > 0:
            time.sleep(_CONNECT_RETRY_DELAY_SECONDS)
        try:
            pool = _get_pool()
            with pool.connection() as conn:
                try:
                    yield conn
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise DatabaseUnavailableError(f"Database operation failed: {e}") from e
            return
        except DatabaseUnavailableError:
            # A query/commit failure is not transient -- retrying won't help.
            raise
        except Exception as e:
            last_error = e
            _discard_pool()

    raise DatabaseUnavailableError(
        f"Could not connect to the Postgres database after {_CONNECT_RETRY_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def execute(sql: str, params: tuple = ()) -> None:
    """Runs one write statement (INSERT/UPDATE/DELETE/DDL) and commits."""
    with get_connection() as conn:
        conn.execute(sql, params)


def fetch_all(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with get_connection() as conn:
        return conn.execute(sql, params).fetchall()


def fetch_one(sql: str, params: tuple = ()) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        return conn.execute(sql, params).fetchone()


def execute_rowcount(sql: str, params: tuple = ()) -> int:
    """Runs a write statement and returns how many rows it affected."""
    with get_connection() as conn:
        cursor = conn.execute(sql, params)
        return cursor.rowcount


def execute_returning(sql: str, params: tuple = ()) -> Any:
    """For INSERT ... RETURNING <col> -- returns that column's value."""
    with get_connection() as conn:
        row = conn.execute(sql, params).fetchone()
        return list(row.values())[0] if row else None


# Every table this app's admin-mutable data lives in. CREATE TABLE IF NOT
# EXISTS throughout, so ensure_schema() is safe to call repeatedly.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS deals (
    id SERIAL PRIMARY KEY,
    customer_name TEXT NOT NULL,
    company TEXT NOT NULL,
    use_case TEXT NOT NULL DEFAULT '',
    metrics TEXT NOT NULL DEFAULT '',
    economic_buyer TEXT NOT NULL DEFAULT '',
    decision_criteria TEXT NOT NULL DEFAULT '',
    decision_process TEXT NOT NULL DEFAULT '',
    paper_process TEXT NOT NULL DEFAULT '',
    identify_pain TEXT NOT NULL DEFAULT '',
    champion TEXT NOT NULL DEFAULT '',
    competition TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_recommendation TEXT,
    last_discovery TEXT,
    last_qualification TEXT
);

CREATE TABLE IF NOT EXISTS requirements (
    id SERIAL PRIMARY KEY,
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
    temp_min DOUBLE PRECISION,
    temp_max DOUBLE PRECISION,
    target_gas TEXT NOT NULL DEFAULT '',
    sensing_range TEXT NOT NULL DEFAULT '',
    accuracy TEXT NOT NULL DEFAULT '',
    environmental TEXT NOT NULL DEFAULT '',
    probe_length TEXT NOT NULL DEFAULT '',
    suggested_code TEXT NOT NULL DEFAULT '',
    additional_requirements TEXT NOT NULL DEFAULT '',
    extra_fields TEXT NOT NULL DEFAULT '',
    deal_id INTEGER
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    verdict TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    resolved INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sales_aids (
    id SERIAL PRIMARY KEY,
    created_at TEXT NOT NULL,
    deal_id INTEGER,
    use_case TEXT NOT NULL DEFAULT '',
    compare_against TEXT NOT NULL DEFAULT '',
    result TEXT NOT NULL
);

-- One row per Approved Claims bullet, in display/save order.
CREATE TABLE IF NOT EXISTS approved_claims (
    position INTEGER PRIMARY KEY,
    bullet TEXT NOT NULL
);

-- The free-text paragraph above the bullet list -- a single row.
CREATE TABLE IF NOT EXISTS approved_claims_header (
    id INTEGER PRIMARY KEY DEFAULT 1,
    header TEXT NOT NULL DEFAULT '',
    CONSTRAINT approved_claims_header_single_row CHECK (id = 1)
);

-- One row per Restricted Claims policy entry, in display/save order.
CREATE TABLE IF NOT EXISTS restricted_claims (
    position INTEGER PRIMARY KEY,
    category TEXT NOT NULL,
    keywords TEXT NOT NULL,
    always_unsupported BOOLEAN NOT NULL DEFAULT FALSE,
    note TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS document_types (
    document_name TEXT PRIMARY KEY,
    doc_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS product_field_overrides (
    product_name TEXT PRIMARY KEY,
    data JSONB NOT NULL
);

-- Admin correction of a document's auto-assigned product name, keyed by filename.
CREATE TABLE IF NOT EXISTS product_name_overrides (
    document_name TEXT PRIMARY KEY,
    product_name TEXT NOT NULL
);

-- The full requirements_fields.json list -- a single row.
CREATE TABLE IF NOT EXISTS requirements_fields (
    id INTEGER PRIMARY KEY DEFAULT 1,
    fields JSONB NOT NULL,
    CONSTRAINT requirements_fields_single_row CHECK (id = 1)
);

-- Uploaded document bytes + metadata; local data/approved_docs is rebuilt from this on cold start.
CREATE TABLE IF NOT EXISTS documents (
    document_name TEXT PRIMARY KEY,
    content BYTEA NOT NULL,
    size_bytes INTEGER NOT NULL,
    uploaded_at TEXT NOT NULL
);
"""

_schema_ensured = False


def ensure_schema() -> None:
    """Creates every table if missing. Runs at most once per process --
    a fast-path guard, not required for correctness."""
    global _schema_ensured
    if _schema_ensured:
        return
    with get_connection() as conn:
        conn.execute(SCHEMA_SQL)
    _schema_ensured = True
