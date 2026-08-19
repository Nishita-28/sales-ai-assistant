"""Durable storage for approved-document *bytes* in Postgres (Neon) --
the piece that makes an admin-uploaded document survive a Streamlit
Community Cloud redeploy or cold start, since the ephemeral local
filesystem (data/approved_docs/) does not.

Only meaningful when app.db.is_postgres_enabled() -- there's no local-
file equivalent to fall back to here, because the local file already IS
the storage in that mode (admin_page.py writes it directly, same as
always). When Postgres is enabled, every write here happens *alongside*
the local file write, not instead of it: the local copy is still what
the document pipeline (python-docx, pypdf, openpyxl, ...) actually reads
from, since none of that code operates on in-memory bytes. Postgres is
the durable backup copy the app rebuilds local disk *from* on cold
start -- see sync_local_docs_from_postgres, called once at app boot
(see app.streamlit_app).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app import db


def save_document_bytes(document_name: str, content: bytes) -> None:
    """Upserts one document's bytes -- called right after the local file
    write in admin_page.py's upload handler, so an add or an overwrite-
    on-edit both stay in sync with Postgres."""
    db.ensure_schema()
    with db.get_connection() as conn:
        conn.execute(
            "INSERT INTO documents (document_name, content, size_bytes, uploaded_at) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (document_name) DO UPDATE SET "
            "content = EXCLUDED.content, size_bytes = EXCLUDED.size_bytes, uploaded_at = EXCLUDED.uploaded_at",
            (document_name, content, len(content), datetime.now().isoformat(timespec="seconds")),
        )


def load_document_bytes(document_name: str) -> bytes | None:
    db.ensure_schema()
    row = db.fetch_one("SELECT content FROM documents WHERE document_name = %s", (document_name,))
    return bytes(row["content"]) if row else None


def delete_document_bytes(document_name: str) -> None:
    """Called right after a document is moved to data/removed_docs/ --
    Postgres should never keep serving bytes for a document the admin
    just removed from the knowledge base."""
    db.ensure_schema()
    db.execute("DELETE FROM documents WHERE document_name = %s", (document_name,))


def list_stored_documents() -> list[str]:
    db.ensure_schema()
    rows = db.fetch_all("SELECT document_name FROM documents ORDER BY document_name ASC")
    return [row["document_name"] for row in rows]


def sync_local_docs_from_postgres(docs_dir: str | Path = "data/approved_docs") -> int:
    """Pulls every stored document down into docs_dir, overwriting
    whatever's there -- Postgres is the durable source of truth, so this
    makes local disk match it exactly. Called once at app boot (a cold
    start's local data/approved_docs/ is either empty or leftover from a
    previous, unrelated container). Returns how many files were written.

    Deliberately does NOT delete a local file that has no matching
    Postgres row -- the only way that happens is a document added
    through some path that bypassed save_document_bytes, and silently
    deleting local content on a mismatch is a worse failure mode than
    leaving an extra file around."""
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    db.ensure_schema()
    rows = db.fetch_all("SELECT document_name, content FROM documents")
    for row in rows:
        (docs_dir / row["document_name"]).write_bytes(bytes(row["content"]))
    return len(rows)
