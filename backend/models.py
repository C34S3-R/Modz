"""Shared pipeline constants and JSON-facing record helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List


STAGE_SPECS = (
    {"key": "validation", "label": "Video Validation"},
    {"key": "extraction", "label": "Frame Extraction"},
    {"key": "filtering", "label": "Frame Filtering"},
    {"key": "preprocessing", "label": "Preprocessing"},
    {"key": "meshroom", "label": "Meshroom Reconstruction"},
    {"key": "mesh_validation", "label": "Mesh Validation"},
    {"key": "blender", "label": "Blender Processing"},
    {"key": "lights", "label": "Light Analysis"},
    {"key": "light_creation", "label": "Light Creation"},
    {"key": "light_validation", "label": "Light Validation"},
    {"key": "bussid_prep", "label": "Export Preparation (BUSSID optional)"},
    {"key": "export", "label": "Export"},
    {"key": "package", "label": "Package"},
)

STAGE_KEYS = tuple(stage["key"] for stage in STAGE_SPECS)
STAGE_BY_KEY = {stage["key"]: stage for stage in STAGE_SPECS}

DEFAULT_OPTIONS: Dict[str, Any] = {
    "photogrammetry": True,
    "preprocessing": False,
    "lights": False,
    "blender": True,
    "bussid": False,
    "mask_background": True,
    "fps": 4,
    "max_frames": 300,
    "force_weak_source": False,
    "image_format": "jpg",
    "resolution": "1920x1080",
    "quality": 2,
}

# Project kinds.  The photogrammetry pipeline underneath is identical for
# all of them; the kind only selects sensible defaults and, for "vehicle",
# keeps the BUSSID game-export path switched on.
PROJECT_TYPES = ("general", "vehicle")

# Creation presets for small first models.  Applied underneath any
# explicitly supplied option, so a user can still fine-tune afterwards.
PRESET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "small": {"fps": 2, "max_frames": 120},
    "standard": {"fps": 4, "max_frames": 300},
    "vehicle": {"fps": 4, "max_frames": 400},
}
PRESETS = tuple(PRESET_DEFAULTS)

PHOTOGRAMMETRY_STAGES = {
    "extraction",
    "filtering",
    "preprocessing",
    "meshroom",
    "mesh_validation",
}
LIGHT_STAGES = {"lights", "light_creation", "light_validation"}
BUSSID_STAGES = {"bussid_prep", "export", "package"}


def now() -> float:
    return time.time()


def default_state() -> Dict[str, Any]:
    return {
        "stages": {
            key: {"status": "pending", "updated_at": now()}
            for key in STAGE_KEYS
        },
        "export": {"status": "idle", "progress": 0},
        "input_revision": 1,
    }


def normalize_stage_status(status: str) -> str:
    return "done" if status in {"done", "completed"} else status


def public_stage(spec: Dict[str, str], record: Dict[str, Any]) -> Dict[str, Any]:
    result = record.get("result") or {}
    answer: Dict[str, Any] = {
        "key": spec["key"],
        "label": spec["label"],
        "status": normalize_stage_status(str(record.get("status", "pending"))),
    }
    if isinstance(result, dict):
        for key, value in result.items():
            if key not in {"command", "stdout", "stderr"}:
                answer[key] = value
    if record.get("error"):
        answer["error"] = record["error"]
    return answer


def public_project(
    row: Dict[str, Any],
    files: Iterable[Dict[str, Any]] = (),
    job: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    file_list = list(files)
    answer = {
        "id": row["id"],
        "slug": row.get("slug"),
        "name": row["name"],
        "description": row.get("description", ""),
        "status": row.get("status", "idle"),
        "current_stage": row.get("current_stage"),
        "progress": float(row.get("progress") or 0),
        "options": row.get("options") or {},
        "error": row.get("error"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "storage_root": row.get("root_path") or row.get("storage_root"),
        "revision": int(row.get("revision") or 1),
        "project_type": row.get("project_type") or "general",
        "preset": row.get("preset") or "standard",
        "inputs": file_list,
        "files": file_list,
    }
    if job:
        started = job.get("started_at")
        finished = job.get("finished_at")
        answer.update(
            {
                "job_id": job.get("id"),
                "operation": job.get("operation"),
                "started_at": started,
                "finished_at": finished,
                "elapsed_seconds": (
                    round((finished or time.time()) - started, 1)
                    if started
                    else None
                ),
            }
        )
    return answer


def public_file(row: Dict[str, Any], download_url: str | None = None) -> Dict[str, Any]:
    answer = {
        "id": row.get("id"),
        "name": row.get("filename"),
        "filename": row.get("filename"),
        "original_name": row.get("original_name") or row.get("filename"),
        "path": row.get("path"),
        "kind": row.get("kind"),
        "type": row.get("kind"),
        "mime_type": row.get("mime_type"),
        "size": int(row.get("size") or 0),
        "created_at": row.get("created_at"),
        "revision": int(row.get("revision") or 1),
    }
    if download_url:
        answer["url"] = download_url
        answer["download_url"] = download_url
    return answer


def public_light(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "project_id": row.get("project_id"),
        "name": row.get("name"),
        "category": row.get("category"),
        "type": row.get("category"),
        "position": row.get("position") or [0, 0, 0],
        "rotation": row.get("rotation") or [0, 0, 0],
        "scale": row.get("scale") or [1, 1, 1],
        "material": row.get("material"),
        "object": row.get("object_name") or row.get("name"),
        "status": row.get("status", "Detected"),
        "metadata": row.get("metadata") or {},
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def public_job(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "job_id": row.get("id"),
        "project_id": row.get("project_id"),
        "type": row.get("type"),
        "status": row.get("status"),
        "current_stage": row.get("current_stage"),
        "operation": row.get("operation"),
        "progress": float(row.get("progress") or 0),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "error": row.get("error"),
    }
