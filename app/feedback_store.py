"""Stores Correct/Wrong/Unsafe feedback on generated answers in a local
SQLite database, so the feedback buttons in the UI actually do something.

Backed by Postgres (Neon) when app.db.is_postgres_enabled() -- see
app.deals_store's module docstring for why. count_feedback and
most_reported_question use positional row[0]/row[1] access on the
SQLite side (sqlite3.Row supports that); the Postgres branch below uses
explicit column aliases and key-based access instead, since psycopg's
dict_row only supports the latter.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app import db

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
    now = datetime.now().isoformat(timespec="seconds")
    if db.is_postgres_enabled():
        db.ensure_schema()
        db.execute(
            "INSERT INTO feedback (created_at, question, answer, verdict, note) VALUES (%s, %s, %s, %s, %s)",
            (now, question, answer, verdict, note.strip()),
        )
        return
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO feedback (created_at, question, answer, verdict, note) VALUES (?, ?, ?, ?, ?)",
            (now, question, answer, verdict, note.strip()),
        )


def resolve_feedback(feedback_id: int) -> None:
    """Marks a report resolved -- it stops counting toward active reports
    and 'most reported question', but stays in the database."""
    if db.is_postgres_enabled():
        db.execute("UPDATE feedback SET resolved = 1 WHERE id = %s", (feedback_id,))
        return
    with closing(_connect()) as conn, conn:
        conn.execute("UPDATE feedback SET resolved = 1 WHERE id = ?", (feedback_id,))


def delete_feedback(feedback_id: int) -> None:
    """Permanently removes a report from the database."""
    if db.is_postgres_enabled():
        db.execute("DELETE FROM feedback WHERE id = %s", (feedback_id,))
        return
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))


def list_all_feedback(limit: int = 50) -> list[Any]:
    """Every feedback event (any verdict, resolved or not), newest first --
    for the admin data-management view. Unlike recent_reports(), this isn't
    limited to active wrong/unsafe reports, since correct-marked events can
    just as easily be test data that needs clearing out."""
    if db.is_postgres_enabled():
        db.ensure_schema()
        return db.fetch_all(
            "SELECT id, created_at, question, verdict FROM feedback ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, question, verdict FROM feedback ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return rows


def count_feedback(before: Optional[str] = None) -> int:
    """Total feedback rows, or only those strictly before the given
    YYYY-MM-DD cutoff if given. Used to preview a bulk-clear's impact
    before committing to it."""
    if db.is_postgres_enabled():
        db.ensure_schema()
        if before:
            row = db.fetch_one(
                "SELECT COUNT(*) AS n FROM feedback WHERE created_at::date < %s", (before,)
            )
        else:
            row = db.fetch_one("SELECT COUNT(*) AS n FROM feedback")
        return row["n"] if row else 0
    with closing(_connect()) as conn:
        if before:
            row = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE date(created_at) < ?", (before,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()
    return row[0] if row else 0


def clear_feedback_before(cutoff_date: str) -> int:
    """Permanently deletes every feedback event recorded before the given
    YYYY-MM-DD date -- e.g. to drop stale test data from the accuracy
    chart without losing real, recent feedback. Returns the number of rows
    removed."""
    if db.is_postgres_enabled():
        return db.execute_rowcount("DELETE FROM feedback WHERE created_at::date < %s", (cutoff_date,))
    with closing(_connect()) as conn, conn:
        cursor = conn.execute("DELETE FROM feedback WHERE date(created_at) < ?", (cutoff_date,))
        return cursor.rowcount


def clear_all_feedback() -> int:
    """Permanently deletes every feedback event, resetting the accuracy
    chart to empty. Returns the number of rows removed."""
    if db.is_postgres_enabled():
        return db.execute_rowcount("DELETE FROM feedback")
    with closing(_connect()) as conn, conn:
        cursor = conn.execute("DELETE FROM feedback")
        return cursor.rowcount


def daily_feedback_counts() -> list[Any]:
    """One row per calendar day that has at least one feedback event, with
    correct/wrong/unsafe counts for that day, oldest first. Powers the
    Feedback tab's accuracy-over-time chart. Days with zero events are
    simply absent rather than zero-filled, so an inactive stretch doesn't
    read as a false 0% accuracy dip."""
    if db.is_postgres_enabled():
        db.ensure_schema()
        return db.fetch_all(
            """
            SELECT
                created_at::date AS day,
                SUM(CASE WHEN verdict = 'correct' THEN 1 ELSE 0 END) AS correct,
                SUM(CASE WHEN verdict = 'wrong' THEN 1 ELSE 0 END) AS wrong,
                SUM(CASE WHEN verdict = 'unsafe' THEN 1 ELSE 0 END) AS unsafe
            FROM feedback
            GROUP BY day
            ORDER BY day ASC
            """
        )
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                date(created_at) AS day,
                SUM(CASE WHEN verdict = 'correct' THEN 1 ELSE 0 END) AS correct,
                SUM(CASE WHEN verdict = 'wrong' THEN 1 ELSE 0 END) AS wrong,
                SUM(CASE WHEN verdict = 'unsafe' THEN 1 ELSE 0 END) AS unsafe
            FROM feedback
            GROUP BY day
            ORDER BY day ASC
            """
        ).fetchall()
    return rows


def most_reported_question() -> Optional[tuple[str, int]]:
    """Returns (question, report_count) for the unresolved question most
    often marked wrong or unsafe, or None if nothing is currently reported."""
    if db.is_postgres_enabled():
        db.ensure_schema()
        row = db.fetch_one(
            """
            SELECT question, COUNT(*) AS report_count
            FROM feedback
            WHERE verdict IN ('wrong', 'unsafe') AND resolved = 0
            GROUP BY question
            ORDER BY report_count DESC, MAX(created_at) DESC
            LIMIT 1
            """
        )
        return (row["question"], row["report_count"]) if row else None
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


def recent_reports(limit: int = 20) -> list[Any]:
    """Most recent unresolved wrong/unsafe reports, newest first, for admin
    review."""
    if db.is_postgres_enabled():
        db.ensure_schema()
        return db.fetch_all(
            """
            SELECT id, created_at, question, answer, verdict, note
            FROM feedback
            WHERE verdict IN ('wrong', 'unsafe') AND resolved = 0
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
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
