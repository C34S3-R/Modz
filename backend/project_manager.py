"""Project creation, persistence, status, and checkpoint operations."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import HTTPException, UploadFile

from database import Database
from file_manager import FileManager
from models import (
    DEFAULT_OPTIONS,
    PRESETS,
    PRESET_DEFAULTS,
    PROJECT_TYPES,
    STAGE_BY_KEY,
    STAGE_KEYS,
    STAGE_SPECS,
    default_state,
    public_project,
    public_stage,
)
from utils.filesystem import FileValidationError
from utils.logger import ProjectLogger
from utils.validation import as_bool, clean_project_name, parse_json_object


class ProjectManager:
    def __init__(
        self,
        db: Database,
        config: Any,
        logger: ProjectLogger,
        file_manager: Optional[FileManager] = None,
    ):
        self.db = db
        self.config = config
        self.logger = logger
        self.files = file_manager or FileManager(db, config, logger)
        self._lifecycle_lock = threading.RLock()
        self._deleting_projects: set[int] = set()

    @property
    def file_manager(self) -> FileManager:
        return self.files

    def _slugify(self, name: str) -> str:
        value = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return value or "project"

    def _unique_slug(self, name: str) -> str:
        base = self._slugify(name)
        candidate = base
        counter = 2
        while self.db.fetchone("SELECT id FROM projects WHERE slug = ?", (candidate,)):
            candidate = f"{base}_{counter}"
            counter += 1
        return candidate

    def project_dir(self, project: Dict[str, Any]) -> Path:
        root = Path(project.get("storage_root") or self.config.projects_dir)
        return root / project["slug"]

    def _create_directories(self, slug: str) -> Path:
        # Every stage gets its own directory.  This separation is what makes
        # retries and inspection possible without overwriting original input.
        project_dir = self.config.projects_dir / slug
        for name in (
            "input",
            "metadata",
            "frames",
            "processed_frames",
            "rejected_frames",
            "reconstruction",
            "blender",
            "lights",
            "bussid",
            "exports",
            "logs",
            "temp",
        ):
            (project_dir / name).mkdir(parents=True, exist_ok=True)
        return project_dir

    async def create_from_upload(
        self,
        meta: Dict[str, Any],
        uploads: Optional[Iterable[UploadFile]] = None,
    ) -> Dict[str, Any]:
        try:
            name = clean_project_name(meta.get("name"))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        description = str(meta.get("description") or "").strip()[:2000]
        project_type = str(meta.get("project_type") or "general").strip().lower()
        if project_type not in PROJECT_TYPES:
            raise HTTPException(status_code=422, detail=f"project_type must be one of {', '.join(PROJECT_TYPES)}")
        preset = str(meta.get("preset") or ("vehicle" if project_type == "vehicle" else "small")).strip().lower()
        if preset not in PRESETS:
            raise HTTPException(status_code=422, detail=f"preset must be one of {', '.join(PRESETS)}")
        options = dict(DEFAULT_OPTIONS)
        # Presets only tune numeric effort settings underneath; an explicit
        # option from the browser always wins over the preset.
        for key, value in PRESET_DEFAULTS[preset].items():
            options[key] = value
        supplied_options = meta.get("options") or {}
        if isinstance(supplied_options, str):
            try:
                supplied_options = parse_json_object(supplied_options, "options")
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not isinstance(supplied_options, dict):
            supplied_options = {}
        if project_type == "vehicle" and "bussid" not in supplied_options:
            options["bussid"] = True
        if project_type == "vehicle" and "mask_background" not in supplied_options:
            options["mask_background"] = False
        for key, value in supplied_options.items():
            if key in options:
                options[key] = value
        # Keep numeric pipeline settings bounded even if supplied by the browser.
        for key, default, minimum, maximum in (
            ("fps", 4, 1, 60),
            ("max_frames", 300, 1, 10000),
            ("quality", 2, 1, 31),
        ):
            try:
                options[key] = max(minimum, min(maximum, int(options[key])))
            except (TypeError, ValueError):
                options[key] = default
        for key in ("photogrammetry", "preprocessing", "lights", "blender", "bussid", "mask_background", "force_weak_source"):
            options[key] = as_bool(options[key], bool(DEFAULT_OPTIONS[key]))

        slug = self._unique_slug(name)
        created_at = time.time()
        project_id = self.db.execute(
            """
            INSERT INTO projects
                (slug, name, description, status, current_stage, progress, options, state,
                 error, created_at, updated_at, root_path, revision, project_type, preset)
            VALUES (?, ?, ?, 'idle', NULL, 0, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
            """,
            (
                slug,
                name,
                description,
                json.dumps(options, separators=(",", ":")),
                json.dumps(default_state(), separators=(",", ":")),
                created_at,
                created_at,
                str(self.config.projects_dir),
                1,
                project_type,
                preset,
            ),
        )
        project = self.get(project_id)
        if not project:
            raise HTTPException(status_code=500, detail="Project creation failed")
        try:
            self._create_directories(slug)
            self.save_state(project_id, self.state(project_id))
            for upload in uploads or ():
                if upload is not None:
                    await self.files.save_upload(project, upload)
            # Online sources: download server-side (SSRF-guarded, size-capped)
            # so a phone-filmed clip hosted elsewhere works like an upload.
            from utils.url_import import URLImportError, extract_url_list, import_urls_for_project

            url_pairs = extract_url_list(meta)[:10]
            if url_pairs:
                try:
                    await import_urls_for_project(
                        project,
                        self.files,
                        url_pairs,
                        max_bytes=self.config.max_upload_bytes,
                    )
                except URLImportError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            # A failed creation must not leave a half-created project visible.
            self.db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            shutil.rmtree(self.project_dir(project), ignore_errors=True)
            raise
        self.logger.append(
            project_id,
            self.project_dir(project),
            stage="PROJECT",
            operation="create",
            status="completed",
            message=f"Created project {name}",
        )
        return self.get(project_id)  # type: ignore[return-value]

    def _latest_job(
        self,
        project_id: int,
        job_type: str = "pipeline",
    ) -> Optional[Dict[str, Any]]:
        return self.db.fetchone(
            "SELECT * FROM jobs WHERE project_id = ? AND type = ? ORDER BY id DESC LIMIT 1",
            (project_id, job_type),
        )

    def get(self, project_id: int) -> Optional[Dict[str, Any]]:
        row = self.db.fetchone("SELECT * FROM projects WHERE id = ?", (project_id,))
        if not row:
            return None
        project = public_project(
            row,
            self.files.rows_for_project(project_id),
            self._latest_job(project_id),
        )
        project["files"] = self.files.grouped(project_id)
        return project

    def get_or_404(self, project_id: int) -> Dict[str, Any]:
        project = self.get(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    def list(self) -> List[Dict[str, Any]]:
        rows = self.db.fetchall("SELECT * FROM projects ORDER BY updated_at DESC, id DESC")
        projects = []
        for row in rows:
            project = public_project(
                row,
                self.files.rows_for_project(row["id"]),
                self._latest_job(row["id"]),
            )
            project["files"] = self.files.grouped(row["id"])
            projects.append(project)
        return projects

    def is_deleting(self, project_id: int) -> bool:
        with self._lifecycle_lock:
            return int(project_id) in self._deleting_projects

    def delete(self, project_id: int) -> None:
        with self._lifecycle_lock:
            if int(project_id) in self._deleting_projects:
                raise HTTPException(status_code=409, detail="Project deletion is already in progress")
            self._deleting_projects.add(int(project_id))
        try:
            self._delete_project(project_id)
        finally:
            with self._lifecycle_lock:
                self._deleting_projects.discard(int(project_id))

    def _delete_project(self, project_id: int) -> None:
        project = self.get_or_404(project_id)
        if self.db.fetchone(
            "SELECT id FROM jobs WHERE project_id = ? AND status IN ('queued','running','paused','cancelling') LIMIT 1",
            (project_id,),
        ):
            raise HTTPException(status_code=409, detail="Cancel active processing before deleting a project")
        for export in self.db.fetchall(
            "SELECT filename FROM files WHERE project_id = ? AND kind = 'export'",
            (project_id,),
        ):
            mirrored = self.config.output_dir / Path(export["filename"]).name
            try:
                mirrored.unlink(missing_ok=True)
            except OSError:
                pass
        project_dir = self.project_dir(project)
        try:
            if project_dir.exists():
                shutil.rmtree(project_dir)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="Project files could not be removed; project metadata was preserved",
            ) from exc
        self.db.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    def state(self, project_id: int) -> Dict[str, Any]:
        row = self.db.fetchone("SELECT state FROM projects WHERE id = ?", (project_id,))
        if not row:
            return default_state()
        state = row.get("state") or default_state()
        if not isinstance(state, dict):
            state = default_state()
        state.setdefault("stages", {})
        if not isinstance(state["stages"], dict):
            state["stages"] = {}
        state.setdefault("export", {"status": "idle", "progress": 0})
        for key in STAGE_KEYS:
            state["stages"].setdefault(key, {"status": "pending"})
        return state

    def save_state(self, project_id: int, state: Dict[str, Any]) -> None:
        # SQLite is authoritative; the JSON mirror is written atomically for a
        # human inspecting a project directory or recovering after a crash.
        encoded = json.dumps(state, separators=(",", ":"))
        self.db.execute(
            "UPDATE projects SET state = ?, updated_at = ? WHERE id = ?",
            (encoded, time.time(), project_id),
        )
        project = self.db.fetchone(
            "SELECT slug, root_path FROM projects WHERE id = ?",
            (project_id,),
        )
        if project:
            root = Path(project.get("root_path") or self.config.projects_dir)
            state_path = root / project["slug"] / "project_state.json"
            try:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = state_path.with_name(
                    f".{state_path.name}.{uuid.uuid4().hex}.tmp"
                )
                temporary.write_text(
                    json.dumps(state, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                temporary.replace(state_path)
            except OSError:
                # The SQLite checkpoint remains authoritative if the mirror is
                # temporarily unavailable.
                pass

    def update_runtime(
        self,
        project_id: int,
        *,
        status: Optional[str] = None,
        current_stage: Any = "__KEEP__",
        progress: Optional[float] = None,
        error: Any = "__KEEP__",
    ) -> None:
        fields: List[str] = []
        values: List[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if current_stage != "__KEEP__":
            fields.append("current_stage = ?")
            values.append(current_stage)
        if progress is not None:
            fields.append("progress = ?")
            values.append(max(0.0, min(100.0, float(progress))))
        if error != "__KEEP__":
            fields.append("error = ?")
            values.append(
                None
                if error is None
                else json.dumps(error, separators=(",", ":"))
            )
        fields.append("updated_at = ?")
        values.append(time.time())
        values.append(project_id)
        self.db.execute(
            f"UPDATE projects SET {', '.join(fields)} WHERE id = ?",
            values,
        )

    def set_stage(
        self,
        project_id: int,
        key: str,
        status: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        if key not in STAGE_BY_KEY:
            raise ValueError(f"Unknown pipeline stage: {key}")
        state = self.state(project_id)
        state["stages"][key] = {
            "status": status,
            "result": result or {},
            "error": error,
            "updated_at": time.time(),
        }
        self.save_state(project_id, state)
        self.db.execute(
            """
            INSERT INTO checkpoints (project_id, stage, status, result, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(project_id, stage) DO UPDATE SET
                status = excluded.status,
                result = excluded.result,
                updated_at = excluded.updated_at
            """,
            (
                project_id,
                key,
                status,
                json.dumps(result or {}, separators=(",", ":")),
                time.time(),
            ),
        )

    def stages(self, project_id: int) -> List[Dict[str, Any]]:
        state = self.state(project_id)
        return [
            public_stage(spec, state["stages"].get(spec["key"], {"status": "pending"}))
            for spec in STAGE_SPECS
        ]

    def status(self, project_id: int) -> Dict[str, Any]:
        row = self.db.fetchone("SELECT * FROM projects WHERE id = ?", (project_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")
        input_files = self.files.rows_for_project(project_id)
        answer = public_project(row, input_files, self._latest_job(project_id))
        answer["stages"] = self.stages(project_id)
        answer["skipped_stages"] = [
            stage["key"] for stage in answer["stages"] if stage["status"] == "skipped"
        ]
        answer["completed_with_skips"] = bool(answer["skipped_stages"])
        answer["package_ready"] = any(
            item.get("kind") == "export"
            and int(item.get("revision") or 1) == int(row.get("revision") or 1)
            for item in self.files.rows_for_project(project_id)
        )
        answer["files"] = self.files.grouped(project_id)
        answer["inputs"] = input_files
        answer["state"] = self.state(project_id)
        answer["export"] = answer["state"].get("export", {"status": "idle", "progress": 0})
        return answer

    def reset_for_retry(self, project_id: int) -> None:
        state = self.state(project_id)
        for key in STAGE_KEYS:
            record = state["stages"].get(key, {})
            if record.get("status") not in {"done", "completed", "skipped"}:
                state["stages"][key] = {"status": "pending", "updated_at": time.time()}
        self.save_state(project_id, state)
        self.update_runtime(
            project_id,
            status="queued",
            current_stage=None,
            progress=self._completed_progress(state),
            error=None,
        )

    def _completed_progress(self, state: Dict[str, Any]) -> float:
        completed = sum(
            1
            for record in state.get("stages", {}).values()
            if record.get("status") in {"done", "completed", "skipped"}
        )
        return round(completed / max(1, len(STAGE_KEYS)) * 100, 2)

    def note_input_change(self, project_id: int, reason: str = "inputs changed") -> Dict[str, Any]:
        # A new upload/option set makes old generated artifacts stale.  Keep
        # those files for recovery, but reset checkpoints so a retry rebuilds
        # from the new revision.
        project = self.get_or_404(project_id)
        revision = int(project.get("revision") or 1) + 1
        self.db.execute(
            "UPDATE projects SET revision = ?, updated_at = ? WHERE id = ?",
            (revision, time.time(), project_id),
        )
        state = self.state(project_id)
        state["stages"] = {
            key: {"status": "pending", "updated_at": time.time()}
            for key in STAGE_KEYS
        }
        state["export"] = {"status": "idle", "progress": 0}
        state["input_revision"] = revision
        self.save_state(project_id, state)
        self.db.execute("DELETE FROM checkpoints WHERE project_id = ?", (project_id,))
        self.update_runtime(
            project_id,
            status="idle",
            current_stage=None,
            progress=0,
            error=None,
        )
        self.logger.append(
            project_id,
            self.project_dir(project),
            stage="PROJECT",
            operation="invalidate",
            status="completed",
            message=f"Invalidated pipeline checkpoints: {reason}",
        )
        return self.get_or_404(project_id)

    def update_options(self, project_id: int, values: Dict[str, Any]) -> Dict[str, Any]:
        if values:
            self.note_input_change(project_id, "pipeline options changed")
        project = self.get_or_404(project_id)
        options = dict(project.get("options") or {})
        # Backfill defaults added after the project was created.
        for key, default in DEFAULT_OPTIONS.items():
            options.setdefault(key, default)
        for key, value in values.items():
            if key not in options:
                continue
            if key in {"photogrammetry", "preprocessing", "lights", "blender", "bussid", "mask_background", "force_weak_source"}:
                value = as_bool(value, bool(options[key]))
            elif key in {"fps", "max_frames", "quality"}:
                limits = {"fps": (1, 60), "max_frames": (1, 10000), "quality": (1, 31)}
                minimum, maximum = limits[key]
                try:
                    value = max(minimum, min(maximum, int(value)))
                except (TypeError, ValueError):
                    pass
            options[key] = value
        self.db.execute(
            "UPDATE projects SET options = ?, updated_at = ? WHERE id = ?",
            (json.dumps(options, separators=(",", ":")), time.time(), project_id),
        )
        return self.get_or_404(project_id)

    async def add_files(
        self,
        project_id: int,
        uploads: Iterable[UploadFile],
        *,
        invalidate: bool = True,
    ) -> List[Dict[str, Any]]:
        upload_list = [upload for upload in uploads if upload is not None]
        if not upload_list:
            return []
        project = (
            self.note_input_change(project_id, "new input uploaded")
            if invalidate
            else self.get_or_404(project_id)
        )
        return [await self.files.save_upload(project, upload) for upload in upload_list]

    async def import_urls(
        self,
        project_id: int,
        urls: List[Tuple[str, Optional[str]]],
        *,
        invalidate: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fetch online sources into an existing project's input/ directory."""
        from utils.url_import import (
            URLImportError,
            import_urls_for_project,
            validate_source_url,
        )

        pairs = [(u.strip(), h) for u, h in urls if u and u.strip()][:10]
        if not pairs:
            raise HTTPException(status_code=422, detail="At least one source URL is required")
        # Reject malformed/private URLs BEFORE invalidating checkpoints, so a
        # typo never wipes completed stages.
        for url, _hint in pairs:
            try:
                validate_source_url(url)
            except URLImportError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        project = (
            self.note_input_change(project_id, "online source imported")
            if invalidate
            else self.get_or_404(project_id)
        )
        try:
            return await import_urls_for_project(
                project,
                self.files,
                pairs,
                max_bytes=self.config.max_upload_bytes,
            )
        except URLImportError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

