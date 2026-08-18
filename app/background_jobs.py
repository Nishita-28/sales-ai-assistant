"""Runs a slow document-management operation (remove/add/rebuild -- see
admin_page.py) on a background thread, so the admin who triggered it isn't
stuck on a blocking spinner and can navigate to another page immediately.

Streamlit has no push-update mechanism between a background thread and a
browser session, and st.session_state is bound to the session that created
it -- a background thread can't reliably write into it. Job status lives
instead in a plain module-level dict, which IS shared across every session
in this process (a single `streamlit run` process serves every browser
tab), so any page's sidebar can poll it. A caller sees a status change only
on its next rerun -- see streamlit_app.py's sidebar fragment, which uses
st.fragment(run_every=...) to poll this on a short timer without
rerunning (or blocking) the rest of the page.

One job at a time, deliberately: two of these running concurrently would
race on the same Chroma index and JSON config files (remove_document_from_
index, rebuild_product_registry, etc. were never written to be safe under
real concurrent writes) -- _LOCK serializes them, and a second job started
while one is still running waits for the first to finish before its own
work begins, rather than running alongside it.
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
        _JOBS[job_id] = JobStatus(label=label)

    def _run() -> None:
        with _LOCK:
            try:
                fn()
            except Exception as e:
                with _STATE_LOCK:
                    _JOBS[job_id] = JobStatus(
                        label=label, status="error", error=str(e), finished_at=time.time()
                    )
                return
        with _STATE_LOCK:
            _JOBS[job_id] = JobStatus(label=label, status="done", finished_at=time.time())

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
