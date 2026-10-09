"""Persistent settings and external-tool resolution."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from database import Database


class SettingsManager:
    ALLOWED_KEYS = {
        "ffmpeg_path",
        "ffprobe_path",
        "meshroom_path",
        "blender_path",
        "project_dir",
        "output_dir",
        "max_concurrent_jobs",
    }

    def __init__(self, db: Database, config: Any):
        self.db = db
        self.config = config
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        defaults = {
            "ffmpeg_path": self.config.ffmpeg_path,
            "ffprobe_path": self.config.ffprobe_path,
            "meshroom_path": self.config.meshroom_path,
            "blender_path": self.config.blender_path,
            "project_dir": str(self.config.projects_dir),
            "output_dir": str(self.config.output_dir),
            "max_concurrent_jobs": self.config.max_concurrent_jobs,
        }
        for key, value in defaults.items():
            if not self.db.fetchone("SELECT key FROM settings WHERE key = ?", (key,)):
                self.db.execute(
                    "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, json.dumps(value), time.time()),
                )
        # Re-apply persisted values after a restart.
        for key in ("ffmpeg_path", "ffprobe_path", "meshroom_path", "blender_path"):
            setattr(self.config, key, str(self.get(key, getattr(self.config, key))))
        self.config.projects_dir = Path(str(self.get("project_dir", self.config.projects_dir))).expanduser()
        self.config.output_dir = Path(str(self.get("output_dir", self.config.output_dir))).expanduser()
        try:
            self.config.max_concurrent_jobs = max(1, min(4, int(self.get("max_concurrent_jobs", self.config.max_concurrent_jobs))))
        except (TypeError, ValueError):
            pass
        self.config.projects_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def all(self) -> Dict[str, Any]:
        rows = self.db.fetchall("SELECT key, value FROM settings ORDER BY key")
        return {row["key"]: row["value"] for row in rows}

    def update(self, values: Dict[str, Any]) -> Dict[str, Any]:
        # Only allowlisted keys are persisted.  This prevents an API caller
        # from adding arbitrary secrets or configuration fields to the server.
        for key, value in values.items():
            if key not in self.ALLOWED_KEYS:
                continue
            if key == "max_concurrent_jobs":
                try:
                    value = max(1, min(4, int(value)))
                except (TypeError, ValueError):
                    value = self.config.max_concurrent_jobs
            elif key in {"project_dir", "output_dir"}:
                raw_value = str(value or "").strip()
                if not raw_value:
                    continue
                value = str(Path(raw_value).expanduser().resolve())
                Path(value).mkdir(parents=True, exist_ok=True)
            else:
                value = str(value or "")
            self.db.execute(
                """
                INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, json.dumps(value), time.time()),
            )
            if key in {"ffmpeg_path", "ffprobe_path", "meshroom_path", "blender_path"}:
                setattr(self.config, key, value)
            elif key == "project_dir":
                self.config.projects_dir = Path(value)
                self.config.projects_dir.mkdir(parents=True, exist_ok=True)
            elif key == "output_dir":
                self.config.output_dir = Path(value)
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
            elif key == "max_concurrent_jobs":
                self.config.max_concurrent_jobs = int(value)
        return self.all()

    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.fetchone("SELECT value FROM settings WHERE key = ?", (key,))
        return row["value"] if row else default

    def tool(self, key: str) -> str | None:
        return self.config.resolve_tool(str(self.get(key, "") or ""))

    def directory(self, key: str, fallback: Path) -> Path:
        value = self.get(key)
        return Path(str(value)).expanduser() if value else fallback
