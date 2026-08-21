"""Tests for db.get_connection()'s retry behavior (app/db.py) -- a
connection pool sitting idle while Neon's compute auto-suspends can hand
back a stale connection (OperationalError, e.g. "server closed the
connection unexpectedly") on the first real use. get_connection() should
recover by retrying rather than surfacing that to the caller.

Uses a fake pool, not a real Neon connection -- these tests must pass
without network access, same as every other test in this repo.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import psycopg

from app import db


class _FakeConnection:
    def __init__(self) -> None:
        self.committed = False

    def execute(self, sql: str, params: tuple = ()):
        cursor = MagicMock()
        cursor.fetchall.return_value = [{"x": 1}]
        cursor.fetchone.return_value = {"x": 1}
        return cursor

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        pass


class _FlakyOnceThenHealthyPool:
    """First connection() call raises psycopg.OperationalError -- a stale
    connection, the exact failure a suspended-then-resumed Neon compute
    produces. Every call after that succeeds, simulating a fresh
    connection replacing the dead one on retry."""

    def __init__(self) -> None:
        self.call_count = 0

    def close(self) -> None:
        pass

    @contextmanager
    def connection(self):
        self.call_count += 1
        if self.call_count == 1:
            raise psycopg.OperationalError("server closed the connection unexpectedly")
        yield _FakeConnection()


def test_recovers_from_a_stale_connection_on_retry(monkeypatch):
    """The actual recovery path: a failed first attempt must not reach
    the caller as an error at all if a later attempt succeeds."""
    fake_pool = _FlakyOnceThenHealthyPool()
    monkeypatch.setattr(db, "_get_pool", lambda: fake_pool)
    monkeypatch.setattr(db, "_CONNECT_RETRY_DELAY_SECONDS", 0)

    result = db.fetch_all("SELECT 1")

    assert result == [{"x": 1}]
    assert fake_pool.call_count == 2, "expected exactly one failed attempt followed by one that succeeded"


def test_discards_the_pool_after_a_failed_attempt(monkeypatch):
    """_discard_pool() must run on a failed attempt -- otherwise a retry
    just asks the same possibly-broken pool object again instead of
    getting a genuinely fresh one."""
    fake_pool = _FlakyOnceThenHealthyPool()
    monkeypatch.setattr(db, "_get_pool", lambda: fake_pool)
    monkeypatch.setattr(db, "_CONNECT_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(db, "_pool", fake_pool)
    monkeypatch.setattr(db, "_schema_ensured", True)

    db.fetch_all("SELECT 1")

    assert db._schema_ensured is False, "a failed attempt should have discarded the pool and reset the schema flag"


def test_a_real_query_error_is_not_retried(monkeypatch):
    """A failure that happens with a live connection (bad SQL, a
    constraint) is a real error, not a connectivity blip -- retrying it
    can't help and shouldn't be attempted."""

    class _AlwaysConnectsButQueryFails:
        def __init__(self) -> None:
            self.connect_count = 0

        def close(self) -> None:
            pass

        @contextmanager
        def connection(self):
            self.connect_count += 1
            conn = _FakeConnection()
            conn.execute = MagicMock(side_effect=psycopg.errors.UndefinedTable("relation does not exist"))
            yield conn

    fake_pool = _AlwaysConnectsButQueryFails()
    monkeypatch.setattr(db, "_get_pool", lambda: fake_pool)
    monkeypatch.setattr(db, "_CONNECT_RETRY_DELAY_SECONDS", 0)

    try:
        db.fetch_all("SELECT * FROM nonexistent")
        assert False, "expected DatabaseUnavailableError"
    except db.DatabaseUnavailableError:
        pass

    assert fake_pool.connect_count == 1, "a real query error should fail immediately, not retry"
