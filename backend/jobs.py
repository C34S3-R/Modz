"""Compatibility names for the old in-memory job placeholder.

Production job state is persisted by :class:`backend.database.Database` and
managed by :class:`backend.job_manager.JobManager`.  These functions remain
only as a clear failure point for old imports; new code must not create a
second, process-local registry.
"""

from __future__ import annotations

from typing import Any


def create(filename: str) -> dict[str, Any]:
    raise RuntimeError("The legacy in-memory jobs registry was removed; use JobManager.enqueue_pipeline()")


def get(job_id: Any) -> None:
    raise RuntimeError("The legacy in-memory jobs registry was removed; use JobManager.get_job()")


def set_status(job_id: Any, status: str) -> None:
    raise RuntimeError("The legacy in-memory jobs registry was removed; use JobManager")


def list_all() -> list[dict[str, Any]]:
    raise RuntimeError("The legacy in-memory jobs registry was removed; use JobManager.list_jobs()")
