"""Reads/writes data/approved_claims.md as a header plus a flat bullet
list, so the Admin UI can list/add/edit/delete individual claims without
hand-editing markdown. The file follows a simple shape: a title, an intro
paragraph, then one "- " bullet per line -- this module doesn't assume
anything more specific than that.

Only approved_claims.md uses this -- it's reference documentation with no
code dependency. data/restricted_claims.yaml (what app/claim_checker.py
actually enforces) uses the structured loader in app/restricted_policy.py
instead; a flat bullet list can't carry the category/keywords/
always-unsupported fields enforcement needs.
"""
from __future__ import annotations

from pathlib import Path


def load_claims(path: Path) -> tuple[str, list[str]]:
    """Splits a claims file into (header, bullets). header is everything
    before the first top-level "- " bullet line, kept verbatim; each
    bullet is one line's text with its leading "- " stripped. Returns
    ("", []) if the file doesn't exist yet."""
    if not path.exists():
        return "", []

    lines = path.read_text(encoding="utf-8").splitlines()
    header_lines: list[str] = []
    bullets: list[str] = []
    in_bullets = False

    for line in lines:
        if line.startswith("- "):
            in_bullets = True
            bullets.append(line[2:])
        elif not in_bullets:
            header_lines.append(line)

    return "\n".join(header_lines).rstrip("\n"), bullets


def save_claims(path: Path, header: str, bullets: list[str]) -> None:
    """Writes the header back verbatim, followed by one "- " bullet per
    non-empty entry. Blank/whitespace-only bullets are dropped, so a row
    cleared in the editor is effectively a delete."""
    lines = [header.rstrip("\n"), ""]
    for bullet in bullets:
        text = bullet.strip()
        if text:
            lines.append(f"- {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
