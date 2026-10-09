"""Server status, settings, and global log routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from api.helpers import json_body

router = APIRouter(prefix="/api", tags=["system"])


def _services(request: Request):
    return request.app.state.services


@router.get("/status")
@router.get("/system/status")
async def status(request: Request):
    services = _services(request)
    result = await run_in_threadpool(
        services.system.status,
        services.config.tool_status(),
    )
    result["settings"] = services.settings.all()
    return result


@router.get("/settings")
async def get_settings(request: Request):
    return _services(request).settings.all()


@router.put("/settings")
async def save_settings(request: Request):
    try:
        data = await json_body(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    services = _services(request)
    if any(key in data for key in {"project_dir", "output_dir"}) and services.db.fetchone(
        "SELECT id FROM jobs WHERE status IN ('queued','running','paused','cancelling') LIMIT 1"
    ):
        raise HTTPException(
            status_code=409,
            detail="Wait for active jobs to finish before changing storage directories",
        )
    result = services.settings.update(data)
    services.jobs.reconfigure()
    return result


@router.post("/settings")
async def save_settings_post(request: Request):
    return await save_settings(request)


@router.get("/system/logs")
async def system_logs(request: Request, project_id: int | None = Query(None)):
    services = _services(request)
    if project_id is not None:
        services.projects.get_or_404(project_id)
        return services.logger.project_logs(project_id)
    path = services.config.logs_dir / "server.log"
    if not path.is_file():
        return []
    try:
        return [
            {"time": None, "message": line.rstrip(), "level": "INFO"}
            for line in path.read_text(errors="replace").splitlines()[-500:]
        ]
    except OSError:
        return []


@router.get("/logs")
async def get_logs(request: Request):
    return await system_logs(request)


@router.delete("/logs")
async def clear_logs(request: Request):
    services = _services(request)
    services.logger.clear()
    return {"cleared": True}


@router.get("/logs/download")
async def download_logs(request: Request):
    services = _services(request)
    path = services.config.logs_dir / "server.log"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No server log exists yet")
    return FileResponse(path, filename="server.log", media_type="text/plain")
