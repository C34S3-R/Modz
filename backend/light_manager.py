"""Light metadata and analysis services."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import HTTPException

from database import Database
from models import public_light
from utils.logger import ProjectLogger


_LIGHT_PATTERNS = {
    "HEADLIGHTS": ("headlight", "head_lamp", "headlamp"),
    "TAIL_LIGHTS": ("taillight", "tail_lamp", "tail_light", "taillamp"),
    "BRAKE_LIGHTS": ("brake",),
    "INDICATORS": ("indicator", "turn_signal", "turnsignal"),
    "REVERSE": ("reverse",),
    "FOG": ("fog",),
    "DRL": ("drl", "daytime_running", "daylight"),
}

_MATERIALS = {
    "HEADLIGHTS": "LightWhite",
    "TAIL_LIGHTS": "LuzRoja",
    "BRAKE_LIGHTS": "LuzRoja",
    "INDICATORS": "LuzAnaranjada",
    "REVERSE": "LightWhite",
    "FOG": "LightWhite",
    "DRL": "LightWhite",
}


class LightManager:
    def __init__(self, db: Database, logger: ProjectLogger):
        self.db = db
        self.logger = logger

    def rows(self, project_id: int) -> List[Dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM lights WHERE project_id = ? ORDER BY name ASC",
            (project_id,),
        )

    def list(self, project_id: int) -> List[Dict[str, Any]]:
        return [public_light(row) for row in self.rows(project_id)]

    def get(self, project_id: int, light_id: Any) -> Optional[Dict[str, Any]]:
        if str(light_id).isdigit():
            row = self.db.fetchone(
                "SELECT * FROM lights WHERE project_id = ? AND id = ?",
                (project_id, int(light_id)),
            )
        else:
            row = self.db.fetchone(
                "SELECT * FROM lights WHERE project_id = ? AND name = ?",
                (project_id, str(light_id)),
            )
        return public_light(row) if row else None

    def _clean_values(self, data: Dict[str, Any], existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        existing = existing or {}
        name = str(data.get("name") or existing.get("name") or "").strip()
        if not name or len(name) > 120 or any(c in name for c in "\x00\r\n/\\"):
            raise HTTPException(status_code=422, detail="A safe light name is required")
        category = str(data.get("category") or existing.get("category") or "CUSTOM").upper()
        material = str(data.get("material") or existing.get("material") or "LightWhite")

        def vector(key: str, default: List[float]) -> List[float]:
            value = data.get(key, existing.get(key, default))
            if not isinstance(value, (list, tuple)):
                return list(default)
            result = []
            for item in list(value)[:3]:
                try:
                    result.append(float(item))
                except (TypeError, ValueError):
                    result.append(default[len(result)])
            while len(result) < 3:
                result.append(default[len(result)])
            return result

        return {
            "name": name,
            "category": category,
            "position": vector("position", [0.0, 0.0, 0.0]),
            "rotation": vector("rotation", [0.0, 0.0, 0.0]),
            "scale": vector("scale", [1.0, 1.0, 1.0]),
            "material": material,
            "object_name": str(data.get("object") or data.get("object_name") or existing.get("object") or name),
            "status": str(data.get("status") or existing.get("status") or "Configured"),
        }

    def create(self, project_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        values = self._clean_values(data)
        timestamp = time.time()
        try:
            light_id = self.db.execute(
                """
                INSERT INTO lights
                    (project_id, name, category, position, rotation, scale, material,
                     object_name, status, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    values["name"],
                    values["category"],
                    json.dumps(values["position"]),
                    json.dumps(values["rotation"]),
                    json.dumps(values["scale"]),
                    values["material"],
                    values["object_name"],
                    values["status"],
                    json.dumps(data.get("metadata") or {}),
                    timestamp,
                    timestamp,
                ),
            )
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                raise HTTPException(status_code=409, detail="A light with that name already exists") from exc
            raise
        self.logger.append(
            project_id,
            None,
            stage="LIGHTS",
            operation="create",
            status="completed",
            message=f"Created light {values['name']}",
        )
        return self.get(project_id, light_id)  # type: ignore[return-value]

    def update(self, project_id: int, light_id: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        existing = self.get(project_id, light_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Light not found")
        values = self._clean_values(data, existing)
        try:
            self.db.execute(
                """
                UPDATE lights SET name = ?, category = ?, position = ?, rotation = ?,
                    scale = ?, material = ?, object_name = ?, status = ?, metadata = ?, updated_at = ?
                WHERE project_id = ? AND id = ?
                """,
                (
                    values["name"],
                    values["category"],
                    json.dumps(values["position"]),
                    json.dumps(values["rotation"]),
                    json.dumps(values["scale"]),
                    values["material"],
                    values["object_name"],
                    values["status"],
                    json.dumps(data.get("metadata") or existing.get("metadata") or {}),
                    time.time(),
                    project_id,
                    existing["id"],
                ),
            )
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                raise HTTPException(status_code=409, detail="A light with that name already exists") from exc
            raise
        return self.get(project_id, existing["id"])  # type: ignore[return-value]

    def delete(self, project_id: int, light_id: Any) -> None:
        light = self.get(project_id, light_id)
        if not light:
            raise HTTPException(status_code=404, detail="Light not found")
        self.db.execute("DELETE FROM lights WHERE project_id = ? AND id = ?", (project_id, light["id"]))
        self.logger.append(
            project_id,
            None,
            stage="LIGHTS",
            operation="delete",
            status="completed",
            message=f"Deleted light {light['name']}",
            level="WARNING",
        )

    def analysis(self, project: Dict[str, Any], model_path: Optional[Path] = None) -> Dict[str, Any]:
        text_parts: List[str] = []
        for item in project.get("inputs", []):
            text_parts.append(str(item.get("path", "")))
        if model_path and model_path.exists():
            try:
                with model_path.open("rb") as handle:
                    text_parts.append(handle.read(2 * 1024 * 1024).decode("utf-8", errors="ignore"))
            except OSError:
                pass
        haystack = "\n".join(text_parts)
        detected: Dict[str, bool] = {}
        found: List[str] = []
        objects: List[Dict[str, str]] = []
        for category, tokens in _LIGHT_PATTERNS.items():
            category_found = False
            for token in tokens:
                pattern = re.compile(
                    rf"[A-Za-z0-9_]*{re.escape(token)}[A-Za-z0-9_]*",
                    re.IGNORECASE,
                )
                matches = []
                for match in pattern.findall(haystack):
                    name = str(match).strip("._- ")
                    if len(name) < 2 or name.lower() in matches:
                        continue
                    matches.append(name.lower())
                    objects.append({"name": name[:80], "category": category})
                if matches:
                    category_found = True
            detected[category] = category_found
            if category_found:
                found.append(category)
        warnings: List[str] = []
        if not found:
            warnings.append(
                "No named light objects were detected; add lights manually after the model is available."
            )
        return {
            "detected": detected,
            "lights": found,
            "objects": objects,
            "warnings": warnings,
            "materials": {category: _MATERIALS[category] for category in found},
        }

    def create_detected(self, project_id: int, analysis: Any) -> List[Dict[str, Any]]:
        if isinstance(analysis, dict):
            candidates = analysis.get("objects") or [
                {"name": category.lower().replace("_lights", "").replace("_light", ""), "category": category}
                for category in analysis.get("lights", [])
            ]
        else:
            candidates = [
                {"name": str(category).lower().replace("_lights", "").replace("_light", ""), "category": str(category).upper()}
                for category in (analysis or [])
            ]
        created: List[Dict[str, Any]] = []
        for candidate in candidates:
            category = str(candidate.get("category") or "CUSTOM").upper()
            name = str(candidate.get("name") or "").strip()
            if not name or self.get(project_id, name):
                continue
            try:
                created.append(
                    self.create(
                        project_id,
                        {
                            "name": name,
                            "category": category,
                            "material": _MATERIALS.get(category, "LightWhite"),
                            "status": "Detected",
                        },
                    )
                )
            except HTTPException:
                continue
        return created
