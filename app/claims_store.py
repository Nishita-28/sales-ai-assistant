"""Reads and writes a claims file as a header plus a flat bullet list, so
the Admin UI can list/add/edit/delete claims without hand-editing
markdown. This is reference documentation only, not enforced policy."""
from __future__ import annotations

from pathlib import Path


def load_claims(path: Path) -> tuple[str, list[str]]:
    """Splits a claims file into (header, bullets). Returns ("", []) if the
    file doesn't exist yet."""
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
    """Writes the header back, followed by one bullet per non-empty entry.
    A blank bullet is dropped, so clearing a row in the editor deletes it."""
    lines = [header.rstrip("\n"), ""]
    for bullet in bullets:
        text = bullet.strip()
        if text:
            lines.append(f"- {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
