"""Reads/writes data/restricted_claims.yaml -- the single source of truth
for restricted-claim enforcement -- and turns it into the lookup
structures app/claim_checker.py actually matches against.

Replaces what used to be hardcoded RESTRICTED_TERM_CATEGORIES /
ALWAYS_UNSUPPORTED_TERMS constants in claim_checker.py. One entry here is
one policy topic: a category, the keyword(s)/phrase(s) that trigger it, whether
matches are always treated as unsupported regardless of retrieved
evidence (see PolicyEntry.always_unsupported), and a human-readable note
for the Admin UI. Multiple entries can share a category -- claim_checker.py
checks categories in the order they first appear across all entries, same
as the old dict-of-lists did.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

# The six risk categories streamlit_app.py's RISK_COLORS renders -- kept
# here (not just implied by whatever's in the file) so the Admin UI's
# category picker can't drift from what the rest of the app recognizes.
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
    """Returns [] if the file doesn't exist yet (a fresh install before an
    admin has added any policy) rather than raising -- callers that need
    to distinguish "no policy" from "policy load failed" can check
    path.exists() themselves first."""
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
    raw = [entry.to_dict() for entry in entries]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=88),
        encoding="utf-8",
    )


def build_lookup(entries: list[PolicyEntry]) -> tuple[dict[str, list[str]], set[str]]:
    """Aggregates entries into the two structures claim_checker.py's
    matching logic needs: category -> all its keywords (in first-seen
    category order, mirroring the old dict-of-lists' checked-in-order
    behaviour), and the flat set of keywords that are always unsupported
    regardless of retrieved evidence."""
    categories: dict[str, list[str]] = {}
    always_unsupported: set[str] = set()

    for entry in entries:
        categories.setdefault(entry.category, [])
        categories[entry.category].extend(entry.keywords)
        if entry.always_unsupported:
            always_unsupported.update(entry.keywords)

    return categories, always_unsupported
