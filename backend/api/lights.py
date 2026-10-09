"""Light CRUD and reference analysis routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from api.helpers import json_body

router = APIRouter(prefix="/api", tags=["lights"])


def _services(request: Request):
    return request.app.state.services


@router.get("/projects/{project_id}/lights")
async def list_lights(project_id: int, request: Request):
    services = _services(request)
    services.projects.get_or_404(project_id)
    return services.lights.list(project_id)


@router.post("/projects/{project_id}/lights")
async def create_light(project_id: int, request: Request):
    services = _services(request)
    services.projects.get_or_404(project_id)
    try:
        data = await json_body(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return services.lights.create(project_id, data)


@router.get("/projects/{project_id}/lights/analysis")
async def light_analysis(project_id: int, request: Request):
    services = _services(request)
    project = services.projects.get_or_404(project_id)
    model = services.projects.file_manager.model_file(project)
    return services.lights.analysis(project, model[1] if model else None)


@router.put("/projects/{project_id}/lights")
async def update_light_without_id(project_id: int, request: Request):
    services = _services(request)
    services.projects.get_or_404(project_id)
    try:
        data = await json_body(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    light_id = data.get("id") or data.get("name")
    if not light_id:
        raise HTTPException(status_code=422, detail="Light id or name is required")
    return services.lights.update(project_id, light_id, data)


@router.put("/projects/{project_id}/lights/{light_id}")
async def update_light(project_id: int, light_id: str, request: Request):
    services = _services(request)
    services.projects.get_or_404(project_id)
    try:
        data = await json_body(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return services.lights.update(project_id, light_id, data)


@router.delete("/projects/{project_id}/lights/{light_id}")
async def delete_light(project_id: int, light_id: str, request: Request):
    services = _services(request)
    services.projects.get_or_404(project_id)
    services.lights.delete(project_id, light_id)
    return {"deleted": True, "id": light_id}
