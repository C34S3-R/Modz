"""Shared pipeline context and stage error types."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from database import Database
from file_manager import FileManager
from project_manager import ProjectManager
from settings_manager import SettingsManager
from utils.logger import ProjectLogger


class StageError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        exit_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.exit_code = exit_code
        self.details = details or {}


# MTL statements that name a texture file.  The filename is the last token on
# the line (earlier tokens are options such as "-s 1 1 1").
_MTL_MAP_KEYS = {
    "map_ka", "map_kd", "map_ks", "map_ns", "map_d", "map_pr", "map_pm",
    "map_ps", "map_ke", "map_bump", "bump", "disp", "decal", "refl",
}


def referenced_sidecars(model: Path) -> List[Path]:
    """Resolve the material/texture chain an OBJ model actually references.

    The model's ``mtllib`` statements name material files beside it, and each
    material's texture-map statements name images beside it again.  Copying
    only the model file leaves it textureless in any consumer that resolves
    references relative to the model's folder (BUSSID does), so stages that
    stage a model must carry its whole referenced chain verbatim.
    """
    try:
        text = model.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    seen = {model.name}
    materials: List[Path] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped == "mtllib" or stripped.startswith("mtllib ")):
            continue
        for token in stripped.split()[1:]:
            candidate = model.parent / token.strip("\"'")
            if candidate.is_file() and candidate.name not in seen:
                seen.add(candidate.name)
                materials.append(candidate)
    sidecars: List[Path] = list(materials)
    for material in materials:
        try:
            mtl_text = material.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in mtl_text.splitlines():
            tokens = line.split()
            if not tokens or tokens[0].lower() not in _MTL_MAP_KEYS:
                continue
            candidate = material.parent / tokens[-1].strip("\"'")
            if candidate.is_file() and candidate.name not in seen:
                seen.add(candidate.name)
                sidecars.append(candidate)
    return sidecars


@dataclass
class PipelineContext:
    project: Dict[str, Any]
    project_manager: ProjectManager
    db: Database
    config: Any
    settings: SettingsManager
    logger: ProjectLogger
    file_manager: FileManager
    control: Any
    job_id: int
    emit: Callable[[Dict[str, Any]], None]
    results: Dict[str, Any] = field(default_factory=dict)

    @property
    def project_id(self) -> int:
        return int(self.project["id"])

    @property
    def project_dir(self) -> Path:
        return self.project_manager.project_dir(self.project)

    @property
    def log_path(self) -> Path:
        return self.project_dir / "logs" / "pipeline.log"

    def tool_log(self, name: str) -> Path:
        return self.project_dir / "logs" / f"{name}.log"

    def check(self) -> None:
        self.control.raise_if_cancelled()
        self.control.wait_if_paused()

    def log(
        self,
        message: str,
        *,
        stage: str = "SYSTEM",
        operation: str = "pipeline",
        status: str = "info",
        level: str = "INFO",
        external_command: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.logger.append(
            self.project_id,
            self.project_dir,
            stage=stage,
            operation=operation,
            status=status,
            message=message,
            level=level,
            external_command=external_command,
            exit_code=exit_code,
        )

    def input_files(self, kind: str) -> List[Tuple[Dict[str, Any], Path]]:
        return self.file_manager.input_files(self.project, kind)

    def input_file(self, kind: str) -> Optional[Tuple[Dict[str, Any], Path]]:
        return self.file_manager.input_file(self.project, kind)

    def model_file(self) -> Optional[Tuple[Dict[str, Any], Path]]:
        return self.file_manager.model_file(self.project)

    def files(self, **kwargs: Any) -> List[Tuple[Dict[str, Any], Path]]:
        return self.file_manager.find_files(self.project, **kwargs)

    def write_json(self, directory: str, filename: str, value: Any) -> Path:
        path = self.project_dir / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def register(self, path: Path, kind: str = "generated") -> Dict[str, Any]:
        return self.file_manager.register_file(self.project, path, kind=kind)
