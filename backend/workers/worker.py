"""Compatibility-friendly worker exports.

The concrete single-host worker lives in :mod:`job_manager` so it can be used
by both the FastAPI composition root and focused tests.
"""

from job_manager import JobControl, JobManager, ResourceGate

__all__ = ["JobControl", "JobManager", "ResourceGate"]
