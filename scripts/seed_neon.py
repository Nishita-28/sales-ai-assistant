"""One-time migration: pushes everything currently in the local data/
directory into Postgres (Neon), so the first Streamlit Cloud deployment
doesn't start from an empty knowledge base and empty history.

Run once, locally, with DATABASE_URL pointed at the Neon project -- not
something the app runs itself on every boot, since re-running this after
real admin edits have landed in Neon would silently overwrite them with
whatever's still sitting in the local files.

    python -m scripts.seed_neon

Safe to re-run before any real Postgres data exists -- every step here is
a full replace of its own table(s).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app import db
from app.claims_store import load_claims, save_claims
from app.document_storage import save_document_bytes
from app.document_types import load_document_types, save_document_types
from app.product_field_overrides import load_overrides as load_field_overrides
from app.product_field_overrides import save_overrides as save_field_overrides
from app.requirements_fields import load_fields as load_requirements_fields
from app.requirements_fields import save_fields as save_requirements_fields
from app.restricted_policy import load_entries, save_entries

APPROVED_DOCS_DIR = Path("data/approved_docs")
APPROVED_CLAIMS_PATH = Path("data/approved_claims.md")
RESTRICTED_CLAIMS_PATH = Path("data/restricted_claims.yaml")


def _require_postgres() -> None:
    if not db.is_postgres_enabled():
        print("DATABASE_URL is not configured -- nothing to seed. Set it in .env first.")
        raise SystemExit(1)


class _read_from_local_files:
    """Every load_*/save_* store function is Postgres-aware, and
    DATABASE_URL is set for this whole script -- so a plain load_*() call
    here would read whatever's already in Postgres, not the local files
    this script exists to seed FROM. Forces db.is_postgres_enabled() to
    report False for its duration so load_*() goes through the local-file
    branch; save_*() still runs after the `with` block ends, with
    Postgres enabled again."""

    def __enter__(self) -> None:
        self._original = db.is_postgres_enabled
        db.is_postgres_enabled = lambda: False

    def __exit__(self, *exc: object) -> None:
        db.is_postgres_enabled = self._original


def seed_documents() -> int:
    if not APPROVED_DOCS_DIR.exists():
        print("No local data/approved_docs/ -- skipping documents.")
        return 0
    count = 0
    for path in sorted(APPROVED_DOCS_DIR.iterdir()):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        save_document_bytes(path.name, path.read_bytes())
        count += 1
        print(f"  {path.name} ({path.stat().st_size} bytes)")
    return count


def seed_approved_claims() -> int:
    with _read_from_local_files():
        header, bullets = load_claims(APPROVED_CLAIMS_PATH)
    if not header and not bullets:
        print("No local approved_claims.md content -- skipping.")
        return 0
    save_claims(APPROVED_CLAIMS_PATH, header, bullets)
    return len(bullets)


def seed_restricted_claims() -> int:
    with _read_from_local_files():
        entries = load_entries(RESTRICTED_CLAIMS_PATH)
    if not entries:
        print("No local restricted_claims.yaml content -- skipping.")
        return 0
    save_entries(RESTRICTED_CLAIMS_PATH, entries)
    return len(entries)


def seed_config() -> None:
    with _read_from_local_files():
        types = load_document_types()
        overrides = load_field_overrides()
        fields = load_requirements_fields()

    if types:
        save_document_types(types)
        print(f"  document_types: {len(types)} entries")

    if overrides:
        save_field_overrides(overrides)
        print(f"  product_field_overrides: {len(overrides)} entries")

    save_requirements_fields(fields)
    print(f"  requirements_fields: {len(fields)} fields")


def _sqlite_rows(path: Path, table: str) -> list[sqlite3.Row]:
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(f"SELECT * FROM {table}").fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def seed_deals() -> int:
    rows = _sqlite_rows(Path("data/deals.db"), "deals")
    if not rows:
        return 0
    columns = rows[0].keys()
    col_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    with db.get_connection() as conn:
        conn.execute("DELETE FROM deals")
        for row in rows:
            conn.execute(f"INSERT INTO deals ({col_list}) VALUES ({placeholders})", tuple(row[c] for c in columns))
        conn.execute("SELECT setval(pg_get_serial_sequence('deals', 'id'), COALESCE((SELECT MAX(id) FROM deals), 1))")
    return len(rows)


def seed_requirements() -> int:
    rows = _sqlite_rows(Path("data/requirements.db"), "requirements")
    if not rows:
        return 0
    columns = rows[0].keys()
    col_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    with db.get_connection() as conn:
        conn.execute("DELETE FROM requirements")
        for row in rows:
            conn.execute(
                f"INSERT INTO requirements ({col_list}) VALUES ({placeholders})", tuple(row[c] for c in columns)
            )
        conn.execute(
            "SELECT setval(pg_get_serial_sequence('requirements', 'id'), COALESCE((SELECT MAX(id) FROM requirements), 1))"
        )
    return len(rows)


def seed_feedback() -> int:
    rows = _sqlite_rows(Path("data/feedback.db"), "feedback")
    if not rows:
        return 0
    columns = rows[0].keys()
    col_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    with db.get_connection() as conn:
        conn.execute("DELETE FROM feedback")
        for row in rows:
            conn.execute(f"INSERT INTO feedback ({col_list}) VALUES ({placeholders})", tuple(row[c] for c in columns))
        conn.execute(
            "SELECT setval(pg_get_serial_sequence('feedback', 'id'), COALESCE((SELECT MAX(id) FROM feedback), 1))"
        )
    return len(rows)


def seed_sales_aids() -> int:
    rows = _sqlite_rows(Path("data/sales_aids.db"), "sales_aids")
    if not rows:
        return 0
    columns = rows[0].keys()
    col_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    with db.get_connection() as conn:
        conn.execute("DELETE FROM sales_aids")
        for row in rows:
            conn.execute(
                f"INSERT INTO sales_aids ({col_list}) VALUES ({placeholders})", tuple(row[c] for c in columns)
            )
        conn.execute(
            "SELECT setval(pg_get_serial_sequence('sales_aids', 'id'), COALESCE((SELECT MAX(id) FROM sales_aids), 1))"
        )
    return len(rows)


def main() -> None:
    _require_postgres()
    db.ensure_schema()

    print("Documents:")
    n_docs = seed_documents()
    print(f"-> {n_docs} documents\n")

    print("Approved Claims:")
    n_claims = seed_approved_claims()
    print(f"-> {n_claims} bullets\n")

    print("Restricted Claims:")
    n_restricted = seed_restricted_claims()
    print(f"-> {n_restricted} entries\n")

    print("Config (document types / product field overrides / requirements fields):")
    seed_config()
    print()

    print("Deals / Requirements / Feedback / Sales Aids history:")
    n_deals = seed_deals()
    n_reqs = seed_requirements()
    n_fb = seed_feedback()
    n_sa = seed_sales_aids()
    print(f"  deals: {n_deals}, requirements: {n_reqs}, feedback: {n_fb}, sales_aids: {n_sa}\n")

    print("Seed complete.")


if __name__ == "__main__":
    main()
