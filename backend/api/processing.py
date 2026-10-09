"""Processing queue, status, logs, legacy jobs, and live event routes."""

from __future__ import annotations

import asyncio
import hmac
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from api.helpers import upload_from_form

router = APIRouter(prefix="/api", tags=["processing"])
ws_router = APIRouter(tags=["live"])


def _services(request: Request):
    return request.app.state.services


@router.post("/projects/{project_id}/process")
async def start_processing(project_id: int, request: Request):
    services = _services(request)
    job = services.jobs.enqueue_pipeline(project_id)
    return {"job_id": job["id"], "job": job, "status": job["status"]}


@router.post("/projects/{project_id}/resume")
async def resume_processing(project_id: int, request: Request):
    services = _services(request)
    active = services.jobs.active_job(project_id)
    if not active:
        raise HTTPException(status_code=404, detail="No active job to resume")
    return {"job_id": active["id"], "job": services.jobs.resume(active["id"])}


@router.post("/projects/{project_id}/pause")
async def pause_processing(project_id: int, request: Request):
    services = _services(request)
    active = services.jobs.active_job(project_id)
    if not active:
        raise HTTPException(status_code=404, detail="No active job to pause")
    return services.jobs.pause(active["id"])


@router.post("/projects/{project_id}/cancel")
async def cancel_processing(project_id: int, request: Request):
    services = _services(request)
    active = services.jobs.active_job(project_id)
    if not active:
        raise HTTPException(status_code=404, detail="No active job to cancel")
    return services.jobs.cancel(active["id"])


@router.post("/projects/{project_id}/retry")
async def retry_processing(project_id: int, request: Request):
    services = _services(request)
    return services.jobs.retry(project_id)


@router.get("/projects/{project_id}/status")
async def project_status(project_id: int, request: Request):
    return _services(request).projects.status(project_id)


@router.get("/projects/{project_id}/logs/download")
async def download_project_logs(project_id: int, request: Request):
    services = _services(request)
    project = services.projects.get_or_404(project_id)
    path = services.projects.project_dir(project) / "logs" / "pipeline.log"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No project log exists yet")
    return FileResponse(path, filename=f"{project['slug']}-pipeline.log", media_type="text/plain")


@router.get("/projects/{project_id}/checkpoints")
async def project_checkpoints(project_id: int, request: Request):
    services = _services(request)
    services.projects.get_or_404(project_id)
    return services.projects.state(project_id)


@router.get("/projects/{project_id}/logs")
async def project_logs(project_id: int, request: Request):
    services = _services(request)
    services.projects.get_or_404(project_id)
    return services.logger.project_logs(project_id)


@router.get("/jobs")
async def list_jobs(request: Request):
    return _services(request).jobs.list_jobs()


@router.get("/jobs/{job_id}")
async def get_job(job_id: int, request: Request):
    return _services(request).jobs.get_job_or_404(job_id)


@router.post("/upload")
async def legacy_upload(request: Request):
    """Compatibility endpoint retained for the original placeholder backend."""

    services = _services(request)
    try:
        upload = await upload_from_form(request)
        name = Path(upload.filename or "uploaded_model").stem
        project = await services.projects.create_from_upload(
            {"name": name, "description": "Created through legacy upload endpoint", "options": {}},
            [upload],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    job = services.jobs.enqueue_pipeline(project["id"])
    return {"job": job, "project": project}


@router.get("/events")
async def events(request: Request):
    services = _services(request)
    queue = await services.broker.subscribe()

    async def stream():
        try:
            yield "retry: 3000\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
        finally:
            await services.broker.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@ws_router.websocket("/ws")
async def websocket_events(websocket: WebSocket):
    services = websocket.app.state.services
    token = services.config.api_token
    origin = websocket.headers.get("origin")
    if origin and "*" not in services.config.cors_origins and origin not in services.config.cors_origins:
        # Same-origin handshakes are always allowed, whichever front door
        # the browser reached us on (loopback, LAN IP, tailnet, tunnel):
        # the Origin authority must match the request's own authority.
        authority = origin.split("://", 1)[-1]
        request_authority = websocket.headers.get("host") or websocket.url.netloc
        if authority != request_authority:
            await websocket.close(code=1008, reason="Origin not allowed")
            return
    if token:
        provided = websocket.headers.get("x-api-token") or websocket.query_params.get("token")
        if not provided or not hmac.compare_digest(provided, token):
            await websocket.close(code=1008, reason="API token required")
            return
    await websocket.accept()
    queue = await services.broker.subscribe()
    try:
        await websocket.send_json({"type": "connected", "status": "online"})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
            await websocket.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await services.broker.unsubscribe(queue)
