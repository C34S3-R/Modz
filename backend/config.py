"""Runtime configuration for the vehicle processing server.

All machine-specific paths are configurable through environment variables.  The
application still keeps the defaults relative to the repository so a fresh
checkout can be started without a separate configuration file.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


def _path_from_env(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _int_from_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


@dataclass
class AppConfig:
    """Filesystem, subprocess, and resource settings used by the backend."""

    base_dir: Path
    uploads_dir: Path
    processing_dir: Path
    output_dir: Path
    projects_dir: Path
    logs_dir: Path
    temp_dir: Path
    database_path: Path
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    meshroom_path: str = "meshroom"
    blender_path: str = "blender"
    max_concurrent_jobs: int = 1
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    max_project_storage_bytes: int = 20 * 1024 * 1024 * 1024
    max_project_files: int = 100
    max_export_bytes: int = 8 * 1024 * 1024 * 1024
    command_timeout_seconds: int = 60 * 60
    require_external_tools: bool = False
    api_token: Optional[str] = None
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5-coder:1.5b"
    cors_origins: tuple[str, ...] = (
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    )

    @classmethod
    def from_env(cls, base_dir: Optional[Path] = None) -> "AppConfig":
        # Resolve paths once at startup.  Routes and workers then use the same
        # directories instead of guessing paths relative to the current shell.
        root = Path(
            base_dir
            or os.getenv("MODZ_BASE_DIR", Path(__file__).resolve().parent.parent)
        ).expanduser().resolve()

        uploads = _path_from_env(root, os.getenv("UPLOAD_DIRECTORY", "uploads"))
        processing = _path_from_env(root, os.getenv("PROCESSING_DIRECTORY", "processing"))
        output = _path_from_env(
            root,
            os.getenv("OUTPUT_DIRECTORY", os.getenv("OUTPUTS_DIRECTORY", "output")),
        )
        projects = _path_from_env(root, os.getenv("PROJECT_DIRECTORY", "projects"))
        logs = _path_from_env(root, os.getenv("LOG_DIRECTORY", "logs"))
        database = _path_from_env(root, os.getenv("DATABASE_PATH", "server.db"))
        temp = _path_from_env(root, os.getenv("TEMP_DIRECTORY", str(processing / "_tmp")))

        origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:8000,http://127.0.0.1:8000,http://localhost:8080,http://127.0.0.1:8080",
            ).split(",")
            if origin.strip()
        ) or (
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        )

        return cls(
            base_dir=root,
            uploads_dir=uploads,
            processing_dir=processing,
            output_dir=output,
            projects_dir=projects,
            logs_dir=logs,
            temp_dir=temp,
            database_path=database,
            ffmpeg_path=os.getenv("FFMPEG_PATH", "ffmpeg"),
            ffprobe_path=os.getenv("FFPROBE_PATH", "ffprobe"),
            meshroom_path=os.getenv("MESHROOM_PATH", "meshroom"),
            blender_path=os.getenv("BLENDER_PATH", "blender"),
            max_concurrent_jobs=_int_from_env("MAX_CONCURRENT_JOBS", 1, 1, 4),
            max_upload_bytes=_int_from_env(
                "MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024, 1024 * 1024, 64 * 1024 * 1024 * 1024
            ),
            max_project_storage_bytes=_int_from_env(
                "MAX_PROJECT_STORAGE_BYTES",
                20 * 1024 * 1024 * 1024,
                1024 * 1024,
                1024 * 1024 * 1024 * 1024,
            ),
            max_project_files=_int_from_env("MAX_PROJECT_FILES", 100, 1, 10000),
            max_export_bytes=_int_from_env(
                "MAX_EXPORT_BYTES",
                8 * 1024 * 1024 * 1024,
                1024 * 1024,
                1024 * 1024 * 1024 * 1024,
            ),
            command_timeout_seconds=_int_from_env(
                "COMMAND_TIMEOUT_SECONDS", 60 * 60, 30, 7 * 24 * 60 * 60
            ),
            require_external_tools=os.getenv("REQUIRE_EXTERNAL_TOOLS", "").lower()
            in {"1", "true", "yes", "on"},
            api_token=os.getenv("API_TOKEN") or None,
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b"),
            cors_origins=origins,
        )

    def ensure_directories(self) -> None:
        """Create writable runtime directories before serving requests.

        The database and logs are also runtime data, so they are included here
        even though they are not vehicle pipeline stages.
        """

        for directory in (
            self.uploads_dir,
            self.processing_dir,
            self.output_dir,
            self.projects_dir,
            self.logs_dir,
            self.temp_dir,
            self.database_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_tool(self, configured: Optional[str]) -> Optional[str]:
        """Resolve a configured executable without invoking a shell."""

        if not configured:
            return None
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        return shutil.which(configured)

    def tool_status(self) -> Dict[str, Dict[str, object]]:
        tools = {
            "ffmpeg": self.ffmpeg_path,
            "ffprobe": self.ffprobe_path,
            "meshroom": self.meshroom_path,
            "blender": self.blender_path,
        }
        status: Dict[str, Dict[str, object]] = {}
        for name, configured in tools.items():
            resolved = self.resolve_tool(configured)
            status[name] = {
                "configured": configured,
                "path": resolved,
                "available": bool(resolved),
            }
        return status
