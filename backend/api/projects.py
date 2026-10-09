"""Project CRUD and project creation routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, UploadFile

from api.helpers import project_payload

router = APIRouter(prefix="/api", tags=["projects"])


def _services(request: Request):
    return request.app.state.services


@router.get("/projects")
async def list_projects(request: Request):
    return _services(request).projects.list()


@router.post("/projects", status_code=201)
async def create_project(request: Request):
    try:
        meta, uploads = await project_payload(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return await _services(request).projects.create_from_upload(meta, uploads)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/projects/{project_id}")
async def get_project(project_id: int, request: Request):
    return _services(request).projects.get_or_404(project_id)


@router.put("/projects/{project_id}/options")
async def update_project_options(project_id: int, request: Request):
    """Update pipeline options (e.g. force_weak_source override).

    Changing options invalidates pipeline checkpoints, so it is refused
    while a job is active - same rule as uploads.
    """
    services = _services(request)
    services.projects.get_or_404(project_id)
    if services.projects.is_deleting(project_id):
        raise HTTPException(status_code=409, detail="Project deletion is in progress")
    if services.db.fetchone(
        "SELECT id FROM jobs WHERE project_id = ? AND status IN ('queued','running','paused','cancelling') LIMIT 1",
        (project_id,),
    ):
        raise HTTPException(status_code=409, detail="Wait for active jobs before changing options")
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Request body must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")
    options = payload.get("options", payload)
    if not isinstance(options, dict):
        raise HTTPException(status_code=422, detail="options must be an object")
    return services.projects.update_options(project_id, options)


@router.delete("/projects/{project_id}")
async def delete_project(project_id: int, request: Request):
    _services(request).projects.delete(project_id)
    return {"deleted": True, "id": project_id}
