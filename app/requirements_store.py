"""Stores customer requirement submissions in a local SQLite database, so
a sales rep's technical notes from a call become a clean record for
production instead of a scattered email or notebook entry.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/requirements.db")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requirements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            customer_name TEXT NOT NULL,
            company TEXT NOT NULL,
            application TEXT NOT NULL DEFAULT '',
            install_type TEXT NOT NULL DEFAULT '',
            num_detectors INTEGER NOT NULL DEFAULT 1,
            comm_protocols TEXT NOT NULL DEFAULT '',
            certifications TEXT NOT NULL DEFAULT '',
            installation_area TEXT NOT NULL DEFAULT '',
            hazard_zone TEXT NOT NULL DEFAULT '',
            temp_min REAL,
            temp_max REAL,
            target_gas TEXT NOT NULL DEFAULT '',
            sensing_range TEXT NOT NULL DEFAULT '',
            accuracy TEXT NOT NULL DEFAULT '',
            environmental TEXT NOT NULL DEFAULT '',
            probe_length TEXT NOT NULL DEFAULT '',
            additional_requirements TEXT NOT NULL DEFAULT ''
        )
        """
    )
    return conn


def record_requirement(
    customer_name: str,
    company: str,
    application: str = "",
    install_type: str = "",
    num_detectors: int = 1,
    comm_protocols: str = "",
    certifications: str = "",
    installation_area: str = "",
    hazard_zone: str = "",
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
    target_gas: str = "",
    sensing_range: str = "",
    accuracy: str = "",
    environmental: str = "",
    probe_length: str = "",
    additional_requirements: str = "",
) -> int:
    """Stores one customer requirement submission. Returns the new row id."""
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            """
            INSERT INTO requirements (
                created_at, customer_name, company, application, install_type,
                num_detectors, comm_protocols, certifications, installation_area,
                hazard_zone, temp_min, temp_max, target_gas, sensing_range,
                accuracy, environmental, probe_length, additional_requirements
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"), customer_name, company,
                application, install_type, num_detectors, comm_protocols, certifications,
                installation_area, hazard_zone, temp_min, temp_max, target_gas,
                sensing_range, accuracy, environmental, probe_length, additional_requirements,
            ),
        )
        return cursor.lastrowid


def list_requirements(limit: int = 200) -> list[sqlite3.Row]:
    """Most recent requirement submissions, newest first."""
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM requirements ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()


def delete_requirement(requirement_id: int) -> None:
    """Permanently removes a requirement submission from the database."""
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM requirements WHERE id = ?", (requirement_id,))
