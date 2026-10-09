"""Compatibility shim for the original placeholder module.

The real pipeline now lives in :mod:`backend.pipeline` and is scheduled by
``JobManager``.  Keeping this small import target avoids breaking older local
scripts while ensuring new code does not accidentally run work in an HTTP
request.
"""

from __future__ import annotations

from typing import Any


def process_file(job: dict[str, Any]) -> None:
    """Reject the removed synchronous entry point with an actionable error."""

    raise RuntimeError(
        "The synchronous process_file placeholder was removed; enqueue a project "
        "through JobManager/PipelineRunner instead"
    )
