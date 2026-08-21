"""Reads and writes the Approved Claims list -- a header paragraph plus a
flat bullet list -- so the Admin UI can list/add/edit/delete claims
without hand-editing markdown. Indexed into retrieval, so this content
can inform an answer, but it's not enforced policy the way
restricted_claims.yaml is: it can't, on its own, satisfy a
restricted-category claim (see app.claim_checker.guardrail_source_text).

Backed by Postgres (Neon) when app.db.is_postgres_enabled() -- otherwise
a claim saved to local approved_claims.md would vanish on the next
Streamlit Cloud redeploy. The `path` parameter is only meaningful in the
local-file branch -- kept in both function signatures so every existing
caller works unchanged regardless of which backend is active."""
from __future__ import annotations

from pathlib import Path

from app import db


def load_claims(path: Path) -> tuple[str, list[str]]:
    """Splits the claims list into (header, bullets). Returns ("", [])
    if there's nothing saved yet."""
    if db.is_postgres_enabled():
        db.ensure_schema()
        header_row = db.fetch_one("SELECT header FROM approved_claims_header WHERE id = 1")
        header = header_row["header"] if header_row else ""
        rows = db.fetch_all("SELECT bullet FROM approved_claims ORDER BY position ASC")
        return header, [row["bullet"] for row in rows]

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
    if db.is_postgres_enabled():
        db.ensure_schema()
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO approved_claims_header (id, header) VALUES (1, %s) "
                "ON CONFLICT (id) DO UPDATE SET header = EXCLUDED.header",
                (header.rstrip("\n"),),
            )
            conn.execute("DELETE FROM approved_claims")
            position = 0
            for bullet in bullets:
                text = bullet.strip()
                if text:
                    conn.execute(
                        "INSERT INTO approved_claims (position, bullet) VALUES (%s, %s)",
                        (position, text),
                    )
                    position += 1
        return

    lines = [header.rstrip("\n"), ""]
    for bullet in bullets:
        text = bullet.strip()
        if text:
            lines.append(f"- {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
