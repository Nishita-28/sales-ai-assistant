"""Stores Correct/Wrong/Unsafe feedback on generated answers in a local
SQLite database, so the feedback buttons in the UI actually do something.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/feedback.db")

VALID_VERDICTS = {"correct", "wrong", "unsafe"}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            verdict TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            resolved INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # Migrates databases created before the resolved column existed.
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
    if "resolved" not in existing_columns:
        conn.execute("ALTER TABLE feedback ADD COLUMN resolved INTEGER NOT NULL DEFAULT 0")
    return conn


def record_feedback(question: str, answer: str, verdict: str, note: str = "") -> None:
    """Stores one feedback event. verdict must be 'correct', 'wrong', or 'unsafe'."""
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid verdict '{verdict}' -- expected one of {sorted(VALID_VERDICTS)}.")
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO feedback (created_at, question, answer, verdict, note) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), question, answer, verdict, note.strip()),
        )


def resolve_feedback(feedback_id: int) -> None:
    """Marks a report resolved -- it stops counting toward active reports
    and 'most reported question', but stays in the database."""
    with closing(_connect()) as conn, conn:
        conn.execute("UPDATE feedback SET resolved = 1 WHERE id = ?", (feedback_id,))


def delete_feedback(feedback_id: int) -> None:
    """Permanently removes a report from the database."""
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))


def count_correct() -> int:
    """Total number of answers marked correct."""
    with closing(_connect()) as conn:
        row = conn.execute("SELECT COUNT(*) FROM feedback WHERE verdict = 'correct'").fetchone()
    return row[0] if row else 0


def most_reported_question() -> Optional[tuple[str, int]]:
    """Returns (question, report_count) for the unresolved question most
    often marked wrong or unsafe, or None if nothing is currently reported."""
    with closing(_connect()) as conn:
        row = conn.execute(
            """
            SELECT question, COUNT(*) AS report_count
            FROM feedback
            WHERE verdict IN ('wrong', 'unsafe') AND resolved = 0
            GROUP BY question
            ORDER BY report_count DESC, MAX(created_at) DESC
            LIMIT 1
            """
        ).fetchone()
    return (row[0], row[1]) if row else None


def recent_reports(limit: int = 20) -> list[sqlite3.Row]:
    """Most recent unresolved wrong/unsafe reports, newest first, for admin
    review."""
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, question, answer, verdict, note
            FROM feedback
            WHERE verdict IN ('wrong', 'unsafe') AND resolved = 0
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return rows
