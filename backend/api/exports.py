"""BUSSID export and checklist routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["exports"])


def _services(request: Request):
    return request.app.state.services


@router.get("/projects/{project_id}/export/checklist")
async def export_checklist(project_id: int, request: Request):
    return _services(request).exports.checklist(project_id)


@router.post("/projects/{project_id}/export")
async def build_export(project_id: int, request: Request):
    return _services(request).exports.start(project_id)


@router.get("/projects/{project_id}/export/status")
async def export_status(project_id: int, request: Request):
    return _services(request).exports.status(project_id)


@router.post("/projects/{project_id}/export/cancel")
async def cancel_export(project_id: int, request: Request):
    return _services(request).exports.cancel(project_id)


@router.get("/projects/{project_id}/exports")
async def list_exports(project_id: int, request: Request):
    services = _services(request)
    project = services.projects.get_or_404(project_id)
    return [
        item
        for item in services.projects.file_manager.rows_for_project(project_id)
        if item["kind"] == "export"
    ]
