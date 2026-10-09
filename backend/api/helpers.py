"""Request parsing helpers shared by API routers."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from fastapi import Request
from starlette.datastructures import UploadFile

from utils.validation import parse_json_object


async def project_payload(request: Request) -> Tuple[Dict[str, Any], List[UploadFile]]:
    """Accept both the browser's multipart form and a JSON-only API request."""
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        raw_meta = form.get("meta")
        if raw_meta is not None:
            meta = parse_json_object(raw_meta, "meta")
        else:
            meta = {
                "name": form.get("name"),
                "description": form.get("description") or "",
                "options": form.get("options") or {},
            }
        uploads = [value for value in form.getlist("files") if isinstance(value, UploadFile)]
        return meta, uploads
    try:
        payload = await request.json()
    except Exception as exc:
        raise ValueError("Request body must be JSON or multipart form data") from exc
    if not isinstance(payload, dict):
        raise ValueError("Request body must be an object")
    return payload, []


async def json_body(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise ValueError("Request body must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


async def upload_from_form(request: Request, field: str = "file") -> UploadFile:
    form = await request.form()
    value = form.get(field)
    if not isinstance(value, UploadFile):
        raise ValueError(f"multipart field '{field}' must be a file")
    return value
