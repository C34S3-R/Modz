"""Project file listing, upload, download, deletion, and model metadata routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from api.helpers import project_payload
from utils.filesystem import FileValidationError, resolve_within

router = APIRouter(prefix="/api", tags=["files"])


def _services(request: Request):
    return request.app.state.services


@router.get("/projects/{project_id}/files")
async def list_files(project_id: int, request: Request):
    services = _services(request)
    services.projects.get_or_404(project_id)
    return services.projects.file_manager.grouped(project_id)


@router.post("/projects/{project_id}/upload")
@router.post("/projects/{project_id}/files")
async def upload_project_files(project_id: int, request: Request):
    services = _services(request)
    project = services.projects.get_or_404(project_id)
    if services.projects.is_deleting(project_id):
        raise HTTPException(status_code=409, detail="Project deletion is in progress")
    if services.db.fetchone(
        "SELECT id FROM jobs WHERE project_id = ? AND status IN ('queued','running','paused','cancelling') LIMIT 1",
        (project_id,),
    ):
        raise HTTPException(status_code=409, detail="Wait for active jobs before uploading files")
    try:
        meta, uploads = await project_payload(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not uploads:
        raise HTTPException(status_code=422, detail="At least one file is required")
    # A project-upload request may optionally update its options using the
    # same multipart shape as project creation, but the project name remains
    # stable.
    options_changed = meta.get("options") is not None and isinstance(meta.get("options"), dict)
    if options_changed:
        services.projects.update_options(project_id, meta["options"])
    return await services.projects.add_files(project_id, uploads, invalidate=not options_changed)


@router.get("/projects/{project_id}/files/download")
async def download_project_file(
    project_id: int,
    request: Request,
    path: str = Query(..., min_length=1),
):
    services = _services(request)
    project = services.projects.get_or_404(project_id)
    try:
        _row, resolved = services.projects.file_manager.resolve_file(project, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project file not found") from exc
    return FileResponse(resolved, filename=resolved.name)


@router.post("/projects/{project_id}/import")
async def import_online_sources(project_id: int, request: Request):
    """Fetch reference videos/images from online URLs into the project.

    Body: {"urls": [...]} or {"video_url": ..., "video_urls": [...],
    "image_urls": [...]}.  URLs are SSRF-guarded and size-capped; each
    must resolve to a supported video/image file.
    """
    services = _services(request)
    project = services.projects.get_or_404(project_id)
    if services.projects.is_deleting(project_id):
        raise HTTPException(status_code=409, detail="Project deletion is in progress")
    if services.db.fetchone(
        "SELECT id FROM jobs WHERE project_id = ? AND status IN ('queued','running','paused','cancelling') LIMIT 1",
        (project_id,),
    ):
        raise HTTPException(status_code=409, detail="Wait for active jobs before importing sources")
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Request body must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")
    from utils.url_import import extract_url_list

    pairs = extract_url_list(payload)
    if "urls" in payload and isinstance(payload["urls"], list):
        for item in payload["urls"]:
            if isinstance(item, str) and item.strip() and (item.strip(), None) not in pairs:
                pairs.append((item.strip(), None))
    _ = project
    return await services.projects.import_urls(project_id, pairs)


def _source_check(services, project_id: int):
    from utils.source_check import check_image_set, check_video_source

    project = services.projects.get_or_404(project_id)
    videos = services.projects.file_manager.input_files(project, "video")
    images = services.projects.file_manager.input_files(project, "image")
    if videos:
        _row, path = videos[0]
        report = check_video_source(
            path,
            ffprobe=services.settings.tool("ffprobe_path"),
        )
        report["source"] = "video"
        return report
    if images:
        report = check_image_set([path for _row, path in images])
        report["source"] = "images"
        return report
    return {
        "ok": False,
        "score": "0/0",
        "checks": [],
        "failures": ["no source: upload a video, add images, or import an online URL first"],
        "warnings": [],
        "guidance": "Upload a reference video, add reference images, or import an online source URL first.",
        "meta": {},
        "source": None,
    }


@router.get("/projects/{project_id}/source-check")
async def source_check(project_id: int, request: Request):
    """Report whether the project's video/images meet the quality bar.

    Same gate the validation stage enforces: `ok: false` means processing
    will stop at Video Validation until better sources are supplied.
    """
    return await run_in_threadpool(_source_check, _services(request), project_id)


@router.delete("/projects/{project_id}/files")
async def delete_project_file(
    project_id: int,
    request: Request,
    path: str = Query(..., min_length=1),
):
    services = _services(request)
    project = services.projects.get_or_404(project_id)
    if services.projects.is_deleting(project_id):
        raise HTTPException(status_code=409, detail="Project deletion is in progress")
    if services.db.fetchone(
        "SELECT id FROM jobs WHERE project_id = ? AND status IN ('queued','running','paused','cancelling') LIMIT 1",
        (project_id,),
    ):
        raise HTTPException(status_code=409, detail="Wait for active jobs before deleting files")
    result = services.projects.file_manager.delete(project, path)
    services.projects.note_input_change(project_id, "project file changed")
    return result


@router.get("/projects/{project_id}/files/{file_id}")
async def get_project_file(project_id: int, file_id: int, request: Request):
    services = _services(request)
    project = services.projects.get_or_404(project_id)
    row = services.projects.db.fetchone(
        "SELECT * FROM files WHERE project_id = ? AND id = ?",
        (project_id, file_id),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Project file not found")
    try:
        resolved = resolve_within(services.projects.project_dir(project), row["path"])
    except FileValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Project file no longer exists")
    return FileResponse(resolved, filename=row["filename"])


def _model_info(services, project_id: int):
    project = services.projects.get_or_404(project_id)
    model = services.projects.file_manager.model_file(project)
    if not model:
        return {"url": None, "format": None, "vertices": 0, "faces": 0, "materials": 0, "objects": 0, "file_size": 0}
    row, path = model
    vertices = 0
    faces = 0
    materials = 0
    objects = 0
    if path.suffix.lower() == ".obj":
        try:
            with path.open("r", errors="replace") as handle:
                for line in handle:
                    if line.startswith("v "):
                        vertices += 1
                    elif line.startswith("f "):
                        faces += 1
                    elif line.startswith("mtllib "):
                        materials += 1
                    elif line.startswith("o "):
                        objects += 1
        except OSError:
            pass
    url = row.get("url") or f"/api/projects/{project_id}/files/download?path={row['path']}"
    return {
        "url": url,
        "format": path.suffix.lower().lstrip("."),
        "vertices": vertices,
        "faces": faces,
        "materials": materials,
        "objects": objects,
        "file_size": path.stat().st_size,
    }


@router.get("/projects/{project_id}/model")
async def model_info(project_id: int, request: Request):
    return await run_in_threadpool(_model_info, _services(request), project_id)
