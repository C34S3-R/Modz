"""In-memory job registry.

Tracks the status of processing jobs. Swap the dict for a database or
on-disk store if jobs must survive restarts.
"""

import itertools
import threading
import time
from typing import Optional

_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_ids = itertools.count(1)


def create(filename: str) -> dict:
    """Register a new job for the given filename."""
    job = {
        "id": next(_ids),
        "filename": filename,
        "status": "queued",
        "created_at": time.time(),
    }
    with _lock:
        _jobs[str(job["id"])] = job
    return job


def get(job_id) -> Optional[dict]:
    with _lock:
        job = _jobs.get(str(job_id))
        return dict(job) if job else None


def set_status(job_id, status: str) -> None:
    with _lock:
        job = _jobs.get(str(job_id))
        if job:
            job["status"] = status


def list_all() -> list[dict]:
    with _lock:
        return [dict(job) for job in sorted(_jobs.values(), key=lambda j: j["id"])]
