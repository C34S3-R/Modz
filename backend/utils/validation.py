"""Validation and JSON helpers shared by API and worker code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def parse_json_object(raw: str | bytes | dict[str, Any], field: str) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def clean_project_name(value: Any) -> str:
    name = str(value or "").strip()
    if not name:
        raise ValueError("Project name is required")
    if len(name) > 120:
        raise ValueError("Project name must be 120 characters or fewer")
    if any(character in name for character in "\x00\r\n"):
        raise ValueError("Project name contains an invalid character")
    return name


def parse_number(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
