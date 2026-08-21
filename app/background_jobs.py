"""Runs a slow document-management operation (remove/add/rebuild) on a
background thread, so the admin isn't stuck on a blocking spinner.

Streamlit has no push-update mechanism between a background thread and a
browser session, and st.session_state is bound to its own session, so a
background thread can't write into it. Job status lives instead in a
plain module-level dict shared across every session in this process, so
any page's sidebar can poll it (see streamlit_app.py's fragment, which
uses st.fragment(run_every=...) to poll on a short timer).

One job at a time, deliberately: concurrent jobs would race on the same
Chroma index and JSON config files, which were never written to be safe
under real concurrent writes -- _LOCK serializes them.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_JOBS: dict[str, "JobStatus"] = {}


@dataclass
class JobStatus:
    job_id: str
    label: str
    status: str = "running"  # "running" | "done" | "error"
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


def start_job(job_id: str, label: str, fn: Callable[[], None]) -> None:
    """Starts fn() on a background thread, tracked under job_id/label.
    Overwrites any previous (already-finished) status for the same
    job_id -- a fresh attempt should read as fresh, not show stale
    "done"/"error" text from an earlier run."""
    with _STATE_LOCK:
        _JOBS[job_id] = JobStatus(job_id=job_id, label=label)

    def _run() -> None:
        with _LOCK:
            try:
                fn()
            except Exception as e:
                with _STATE_LOCK:
                    _JOBS[job_id] = JobStatus(
                        job_id=job_id, label=label, status="error", error=str(e), finished_at=time.time()
                    )
                return
        with _STATE_LOCK:
            _JOBS[job_id] = JobStatus(job_id=job_id, label=label, status="done", finished_at=time.time())

    threading.Thread(target=_run, daemon=True).start()


def get_active_jobs() -> list[JobStatus]:
    """Every job still running, plus any that finished (done or error)
    within the last few seconds -- long enough for the sidebar to show
    a brief "Removed X" / "Failed: ..." confirmation before it clears,
    without needing a manual dismiss."""
    cutoff = time.time() - 8
    with _STATE_LOCK:
        return [
            job for job in _JOBS.values()
            if job.status == "running" or (job.finished_at or 0) > cutoff
        ]
