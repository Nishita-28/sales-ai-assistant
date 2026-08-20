"""Postgres (Neon) connection layer -- the durable persistence backing
every admin-mutable store (deals, requirements, feedback, sales aids,
Approved/Restricted Claims, document metadata + bytes, and the small
config JSON stores) when DATABASE_URL is configured.

Streamlit Community Cloud's filesystem is ephemeral: anything written to
local disk (SQLite files, uploaded documents, the Chroma index) is gone
on the next redeploy or cold start. Every store module in this app
checks is_postgres_enabled() and, when true, reads/writes through here
instead of local SQLite/JSON/YAML files -- falling back to each store's
original local-file behavior when DATABASE_URL isn't set, so local
development and tests never need a live Neon connection.

DATABASE_URL is read from st.secrets first, then the environment,
matching app.streamlit_app.get_admin_password's own secrets-then-env
pattern -- use Neon's *pooled* connection string (PgBouncer-backed),
not the direct one, since a Streamlit app can hold several concurrent
connections across sessions/reruns.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class DatabaseUnavailableError(RuntimeError):
    """Raised when DATABASE_URL is configured but the connection (or a
    query against it) fails. Deliberately never swallowed into a silent
    fallback to local files -- that would write an admin's change
    somewhere they don't expect, then quietly lose it on the next
    redeploy without ever telling them. Every store surfaces this as a
    real st.error() in the Admin UI rather than pretending the save
    succeeded."""


def get_database_url() -> Optional[str]:
    """None means "use local SQLite/file storage" -- every store checks
    this before deciding which backend to use. Wrapped in try/except
    because st.secrets raises if no secrets.toml exists at all (the
    normal case for local dev), not just when the one key is missing."""
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
    """A small process-level pool of warm connections, opened lazily on
    first use. A fresh psycopg.connect() per query pays a full TCP/TLS
    handshake to Neon every time (measured at ~0.5s each, even against
    Neon's own pooled endpoint) -- with several queries per page render,
    that overhead dominates page-load time. Reusing pooled connections
    across calls avoids paying it more than once per connection's
    lifetime."""
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
                kwargs={"row_factory": dict_row, "connect_timeout": 20},
                open=True,
                timeout=25,
            )
        except Exception as e:
            raise DatabaseUnavailableError(f"Could not connect to the Postgres database: {e}") from e
    return _pool


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Borrows a connection from the pool, committed on clean exit, rolled
    back and re-raised as DatabaseUnavailableError on failure. The
    connection returns to the pool afterward rather than closing, so the
    next call can reuse it."""
    pool = _get_pool()
    try:
        with pool.connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                if isinstance(e, DatabaseUnavailableError):
                    raise
                raise DatabaseUnavailableError(f"Database operation failed: {e}") from e
    except DatabaseUnavailableError:
        raise
    except Exception as e:
        raise DatabaseUnavailableError(f"Could not connect to the Postgres database: {e}") from e


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
    """Runs one write statement and returns how many rows it affected --
    for a bulk DELETE/UPDATE whose caller reports back how much it did
    (e.g. "cleared 12 old feedback rows")."""
    with get_connection() as conn:
        cursor = conn.execute(sql, params)
        return cursor.rowcount


def execute_returning(sql: str, params: tuple = ()) -> Any:
    """For an INSERT ... RETURNING <col> -- returns that column's value
    from the first (only) returned row."""
    with get_connection() as conn:
        row = conn.execute(sql, params).fetchone()
        return list(row.values())[0] if row else None


# ---------------------------------------------------------------------------
# Schema -- every table this app's admin-mutable data lives in. Created
# once per process on first use (see ensure_schema); CREATE TABLE IF NOT
# EXISTS throughout, so it's always safe to call again.
# ---------------------------------------------------------------------------
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

-- The full requirements_fields.json list -- a single row.
CREATE TABLE IF NOT EXISTS requirements_fields (
    id INTEGER PRIMARY KEY DEFAULT 1,
    fields JSONB NOT NULL,
    CONSTRAINT requirements_fields_single_row CHECK (id = 1)
);

-- Uploaded document bytes + metadata -- the durable source local
-- data/approved_docs is rebuilt from on cold start (see
-- retriever.rebuild_local_docs_from_postgres).
CREATE TABLE IF NOT EXISTS documents (
    document_name TEXT PRIMARY KEY,
    content BYTEA NOT NULL,
    size_bytes INTEGER NOT NULL,
    uploaded_at TEXT NOT NULL
);
"""

_schema_ensured = False


def ensure_schema() -> None:
    """Creates every table if missing. Idempotent; runs at most once per
    process (subsequent calls are a no-op) since every table statement
    is already its own CREATE TABLE IF NOT EXISTS -- the process-level
    guard just avoids a redundant round-trip on every store call."""
    global _schema_ensured
    if _schema_ensured:
        return
    with get_connection() as conn:
        conn.execute(SCHEMA_SQL)
    _schema_ensured = True
