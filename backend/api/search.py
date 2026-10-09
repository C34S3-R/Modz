"""Keyless online search routes (image reference hunt)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search/images")
async def search_images(
    request: Request,
    q: str = Query(..., min_length=2, max_length=200),
    source: str = Query("all"),
    limit: int = Query(20, ge=1, le=30),
):
    """Search free providers (Openverse + Wikimedia Commons) for reference photos.

    No key, no cost.  Returns thumbnails, direct URLs, and license/source
    info; the caller picks winners and POSTs them to /projects/{id}/import.
    """
    from utils.online_search import OnlineSearchError, search_images as run_search

    _ = request
    try:
        return await run_in_threadpool(run_search, q, source=source, limit=limit)
    except OnlineSearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
