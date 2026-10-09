"""AI advisor route (read-only, never runs the pipeline)."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["advisor"])


@router.get("/projects/{project_id}/advisor")
async def project_advisor(project_id: int, request: Request):
    return request.app.state.services.advisor.analyze(project_id)
