"""Reads and writes the restricted-claims policy. One entry is one policy
topic: a category, its trigger keywords, whether matches are always
unsupported, and a note shown in the Admin UI.

Backed by Postgres (Neon) when app.db.is_postgres_enabled() -- this is
what the assistant actually enforces on every answer (see
app.claim_checker.check_restricted_claims), so an edit here silently
reverting on the next Streamlit Cloud redeploy would mean a
certification/pricing/safety guardrail an admin just added quietly
stops being enforced. Falls back to the original local YAML file when
DATABASE_URL isn't set. The `path` parameter is only meaningful in the
local-file branch -- kept in both function signatures so every existing
caller works unchanged regardless of which backend is active.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from app import db

# The risk categories the UI can render -- keeps the Admin category picker
# from drifting out of sync.
KNOWN_CATEGORIES = ["Certification", "Accuracy", "Safety", "Pricing", "Delivery", "Legal"]


class PolicyError(RuntimeError):
    """Wraps YAML load/parse failures for the restricted-claims policy."""


@dataclass
class PolicyEntry:
    category: str
    keywords: list[str]
    always_unsupported: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_entries(path: Path) -> list[PolicyEntry]:
    """Returns [] if there's nothing saved yet, rather than raising."""
    if db.is_postgres_enabled():
        db.ensure_schema()
        rows = db.fetch_all(
            "SELECT category, keywords, always_unsupported, note "
            "FROM restricted_claims ORDER BY position ASC"
        )
        return [
            PolicyEntry(
                category=row["category"],
                keywords=[k.strip() for k in row["keywords"].split(",") if k.strip()],
                always_unsupported=bool(row["always_unsupported"]),
                note=row["note"] or "",
            )
            for row in rows
        ]

    if not path.exists():
        return []

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    except yaml.YAMLError as e:
        raise PolicyError(f"Failed to parse {path}: {e}") from e

    if not isinstance(raw, list):
        raise PolicyError(f"{path} must contain a YAML list of policy entries.")

    entries = []
    for i, item in enumerate(raw):
        try:
            entries.append(
                PolicyEntry(
                    category=item["category"],
                    keywords=list(item.get("keywords") or []),
                    always_unsupported=bool(item.get("always_unsupported", False)),
                    note=item.get("note", ""),
                )
            )
        except (KeyError, TypeError) as e:
            raise PolicyError(f"{path}: entry {i} is malformed ({e}).") from e

    return entries


def save_entries(path: Path, entries: list[PolicyEntry]) -> None:
    if db.is_postgres_enabled():
        db.ensure_schema()
        with db.get_connection() as conn:
            conn.execute("DELETE FROM restricted_claims")
            for position, entry in enumerate(entries):
                conn.execute(
                    "INSERT INTO restricted_claims (position, category, keywords, always_unsupported, note) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (position, entry.category, ", ".join(entry.keywords), entry.always_unsupported, entry.note),
                )
        return

    raw = [entry.to_dict() for entry in entries]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=88),
        encoding="utf-8",
    )


def build_lookup(entries: list[PolicyEntry]) -> tuple[dict[str, list[str]], set[str]]:
    """Groups entries by category -> all its keywords, plus the flat set
    of always-unsupported keywords."""
    categories: dict[str, list[str]] = {}
    always_unsupported: set[str] = set()

    for entry in entries:
        categories.setdefault(entry.category, [])
        categories[entry.category].extend(entry.keywords)
        if entry.always_unsupported:
            always_unsupported.update(entry.keywords)

    return categories, always_unsupported
