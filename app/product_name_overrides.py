"""Admin correction of a document's auto-assigned product name (see
app.chunker.assign_product_name) -- for when the heading the auto-namer
picked isn't the right product name.

Keyed by document filename, not the auto-assigned name itself: the
auto-assigned name is recomputed on every rebuild and can shift, so the
filename is the only stable identity to hang a correction off. Checked
as the final step of name assignment, so an admin's correction survives
every future reindex instead of being silently recomputed away.

Backed by Postgres (Neon) when app.db.is_postgres_enabled()."""
from __future__ import annotations

import json
import time
from pathlib import Path

from app import db

OVERRIDES_PATH = Path("data/product_name_overrides.json")

_CACHE_TTL_SECONDS = 3
_cache: tuple[float, dict[str, str]] | None = None


def load_overrides() -> dict[str, str]:
    """document_name -> admin-chosen product_name."""
    global _cache
    if db.is_postgres_enabled():
        now = time.time()
        if _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
            return _cache[1]
        db.ensure_schema()
        rows = db.fetch_all("SELECT document_name, product_name FROM product_name_overrides")
        result = {row["document_name"]: row["product_name"] for row in rows}
        _cache = (now, result)
        return result

    if not OVERRIDES_PATH.exists():
        return {}
    return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))


def save_overrides(overrides: dict[str, str]) -> None:
    global _cache
    if db.is_postgres_enabled():
        db.ensure_schema()
        with db.get_connection() as conn:
            conn.execute("DELETE FROM product_name_overrides")
            for document_name, product_name in overrides.items():
                conn.execute(
                    "INSERT INTO product_name_overrides (document_name, product_name) VALUES (%s, %s)",
                    (document_name, product_name),
                )
        _cache = None
        return

    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8")


def set_override(document_name: str, product_name: str) -> None:
    overrides = load_overrides()
    overrides[document_name] = product_name
    save_overrides(overrides)


def clear_override(document_name: str) -> None:
    overrides = load_overrides()
    if document_name in overrides:
        del overrides[document_name]
        save_overrides(overrides)
