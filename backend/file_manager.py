"""Upload and project-file management with path traversal protection."""

from __future__ import annotations

import mimetypes
import os
import shutil
import threading
import time
import uuid
from urllib.parse import quote
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from fastapi import HTTPException, UploadFile

from database import Database
from models import public_file
from utils.filesystem import (
    ALLOWED_EXTENSIONS,
    FileValidationError,
    file_kind,
    resolve_within,
    sanitize_filename,
    unique_destination,
    validate_mime_type,
    validate_upload_name,
)
from utils.logger import ProjectLogger


class FileManager:
    def __init__(self, db: Database, config: Any, logger: ProjectLogger):
        self.db = db
        self.config = config
        self.logger = logger
        self._upload_locks: Dict[int, threading.RLock] = {}

    def project_dir(self, project: Dict[str, Any]) -> Path:
        root = Path(project.get("storage_root") or self.config.projects_dir)
        return root / project["slug"]

    async def save_upload(
        self,
        project: Dict[str, Any],
        upload: UploadFile,
    ) -> Dict[str, Any]:
        lock = self._upload_locks.setdefault(int(project["id"]), threading.RLock())
        with lock:
            return await self._save_upload(project, upload)

    async def _save_upload(
        self,
        project: Dict[str, Any],
        upload: UploadFile,
    ) -> Dict[str, Any]:
        """Stream one uploaded file to the project's input directory.

        The temporary ``.part`` file prevents a cancelled request from being
        mistaken for a complete input.  The final filename is sanitized before
        it ever reaches the filesystem.
        """

        try:
            filename, kind = validate_upload_name(upload.filename)
            validate_mime_type(upload.content_type, kind)
        except FileValidationError as exc:
            await upload.close()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        project_dir = self.project_dir(project)
        usage = self.db.fetchone(
            "SELECT COUNT(*) AS file_count, COALESCE(SUM(size), 0) AS total_size FROM files WHERE project_id = ?",
            (project["id"],),
        ) or {"file_count": 0, "total_size": 0}
        if int(usage.get("file_count") or 0) >= self.config.max_project_files:
            await upload.close()
            raise HTTPException(
                status_code=413,
                detail=f"Project file limit ({self.config.max_project_files}) reached",
            )
        storage_used = int(usage.get("total_size") or 0)
        if storage_used >= self.config.max_project_storage_bytes:
            await upload.close()
            raise HTTPException(status_code=413, detail="Project storage quota reached")
        input_dir = project_dir / "input"
        destination = unique_destination(input_dir, filename)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        total = 0
        registered = False
        try:
            with temporary.open("wb") as handle:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.config.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload exceeds the {self.config.max_upload_bytes} byte limit",
                        )
                    if storage_used + total > self.config.max_project_storage_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Project storage quota exceeded",
                        )
                    handle.write(chunk)
            if total == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            os.replace(temporary, destination)
            relative = destination.resolve().relative_to(project_dir.resolve()).as_posix()
            mime_type = upload.content_type or mimetypes.guess_type(destination.name)[0]
            self.db.execute(
                """
                INSERT INTO files
                    (project_id, filename, original_name, path, kind, mime_type, size, created_at,
                     revision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project["id"],
                    destination.name,
                    filename,
                    relative,
                    kind,
                    mime_type,
                    total,
                    time.time(),
                    int(project.get("revision") or 1),
                ),
            )
            registered = True
            self.logger.append(
                project["id"],
                project_dir,
                stage="UPLOAD",
                operation="receive",
                status="completed",
                message=f"Stored {destination.name} ({total} bytes)",
            )
            return self.public_file(self.get_file_by_path(project["id"], relative))
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if not registered:
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        finally:
            await upload.close()

    def register_file(
        self,
        project: Dict[str, Any],
        path: Path,
        *,
        kind: Optional[str] = None,
        original_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        project_dir = self.project_dir(project)
        path = Path(path).resolve()
        relative = path.relative_to(project_dir.resolve()).as_posix()
        try:
            existing = self.get_file_by_path(project["id"], relative)
        except FileNotFoundError:
            existing = None
        stat = path.stat()
        file_kind_value = kind or file_kind(path.name)
        if existing:
            self.db.execute(
                """
                UPDATE files SET filename = ?, original_name = ?, kind = ?, mime_type = ?, size = ?,
                    revision = ?
                WHERE id = ?
                """,
                (
                    path.name,
                    original_name or path.name,
                    file_kind_value,
                    mimetypes.guess_type(path.name)[0],
                    stat.st_size,
                    int(project.get("revision") or 1),
                    existing["id"],
                ),
            )
            return self.public_file(self.get_file_by_path(project["id"], relative))
        self.db.execute(
            """
            INSERT INTO files
                (project_id, filename, original_name, path, kind, mime_type, size, created_at,
                 revision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project["id"],
                path.name,
                original_name or path.name,
                relative,
                file_kind_value,
                mimetypes.guess_type(path.name)[0],
                stat.st_size,
                time.time(),
                int(project.get("revision") or 1),
            ),
        )
        return self.public_file(self.get_file_by_path(project["id"], relative))

    def get_file_by_path(self, project_id: int, path: str) -> Dict[str, Any]:
        row = self.db.fetchone(
            "SELECT * FROM files WHERE project_id = ? AND path = ?",
            (project_id, path),
        )
        if not row:
            raise FileNotFoundError(f"Unknown project file: {path}")
        return self.public_file(row)

    def rows_for_project(self, project_id: int) -> List[Dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT * FROM files WHERE project_id = ? ORDER BY created_at ASC, id ASC",
            (project_id,),
        )
        return [self.public_file(row) for row in rows]

    def raw_rows_for_project(self, project_id: int) -> List[Dict[str, Any]]:
        return self.db.fetchall(
            "SELECT * FROM files WHERE project_id = ? ORDER BY created_at ASC, id ASC",
            (project_id,),
        )

    def public_file(self, row: Dict[str, Any]) -> Dict[str, Any]:
        project_id = row.get("project_id")
        path = row.get("path", "")
        url = f"/api/projects/{project_id}/files/download?path={quote(path, safe='/')}" if project_id else None
        return public_file(row, url)

    def grouped(self, project_id: int) -> List[Dict[str, Any]]:
        labels = {
            "input": "INPUT",
            "frames": "FRAMES",
            "processed_frames": "PROCESSED_FRAMES",
            "rejected_frames": "REJECTED_FRAMES",
            "reconstruction": "RECONSTRUCTION",
            "blender": "BLENDER",
            "lights": "LIGHTS",
            "bussid": "BUSSID",
            "exports": "EXPORTS",
        }
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.rows_for_project(project_id):
            first = item["path"].split("/", 1)[0]
            label = labels.get(first, first.upper() or "FILES")
            groups.setdefault(label, []).append(item)
        return [{"name": name, "files": files} for name, files in groups.items()]

    def resolve_file(self, project: Dict[str, Any], path: str) -> Tuple[Dict[str, Any], Path]:
        # Validate the path before looking up a basename.  This blocks both
        # absolute paths and ../ traversal attempts from becoming downloads.
        project_dir = self.project_dir(project).resolve()
        try:
            resolve_within(project_dir, path)
        except FileValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            row = self.get_file_by_path(project["id"], path)
        except FileNotFoundError:
            # Accept a basename only when it is unambiguous; never accept an
            # arbitrary path outside the registered project files.
            candidates = [
                item
                for item in self.raw_rows_for_project(project["id"])
                if Path(item["path"]).name == Path(path).name
            ]
            if len(candidates) != 1:
                raise
            row = self.public_file(candidates[0])
        project_dir = self.project_dir(project).resolve()
        try:
            resolved = resolve_within(project_dir, row["path"])
        except FileValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="Project file no longer exists")
        return row, resolved

    def delete(self, project: Dict[str, Any], path: str) -> Dict[str, Any]:
        row, resolved = self.resolve_file(project, path)
        if row["kind"] in {"video", "model", "image"} and row["path"].startswith("input/"):
            raise HTTPException(
                status_code=400,
                detail="Original input files are protected; remove them only from the project filesystem after making a backup",
            )
        try:
            resolved.unlink()
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Unable to delete project file") from exc
        self.db.execute("DELETE FROM files WHERE id = ?", (row["id"],))
        self.logger.append(
            project["id"],
            self.project_dir(project),
            stage="FILES",
            operation="delete",
            status="completed",
            message=f"Deleted generated file {row['path']}",
            level="WARNING",
        )
        return {"deleted": True, "path": row["path"]}

    def find_files(
        self,
        project: Dict[str, Any],
        *,
        kinds: Optional[Iterable[str]] = None,
        extensions: Optional[Iterable[str]] = None,
        roots: Optional[Iterable[str]] = None,
        revision: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], Path]]:
        kinds_set = set(kinds or ())
        extensions_set = {str(ext).lower() for ext in (extensions or ())}
        roots_set = {str(root).strip("/") for root in (roots or ())}
        result: List[Tuple[Dict[str, Any], Path]] = []
        project_dir = self.project_dir(project)
        for row in self.raw_rows_for_project(project["id"]):
            if revision is not None and int(row.get("revision") or 1) != revision:
                continue
            if kinds_set and row["kind"] not in kinds_set:
                continue
            if extensions_set and Path(row["filename"]).suffix.lower() not in extensions_set:
                continue
            if roots_set and row["path"].split("/", 1)[0] not in roots_set:
                continue
            try:
                path = resolve_within(project_dir, row["path"])
            except FileValidationError:
                continue
            if path.is_file():
                result.append((self.public_file(row), path))
        return result

    def model_file(self, project: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Path]]:
        priority = [
            {"roots": ["blender"], "extensions": {".blend", ".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply"}},
            {"roots": ["bussid"], "extensions": {".blend", ".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply"}},
            {"roots": ["reconstruction"], "extensions": {".blend", ".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply"}},
            {"roots": ["input"], "extensions": {".blend", ".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply"}},
        ]
        revision = int(project.get("revision") or 1)
        for rule in priority:
            found = self.find_files(
                project,
                extensions=rule["extensions"],
                roots=rule["roots"],
                revision=revision,
            )
            if found:
                return found[0]
        # Databases created before revision tracking can still be recovered.
        for rule in priority:
            found = self.find_files(
                project,
                extensions=rule["extensions"],
                roots=rule["roots"],
            )
            if found:
                return found[0]
        return None

    def input_files(self, project: Dict[str, Any], kind: str) -> List[Tuple[Dict[str, Any], Path]]:
        revision = int(project.get("revision") or 1)
        found = self.find_files(project, kinds=[kind], roots=["input"], revision=revision)
        if not found:
            found = self.find_files(project, kinds=[kind], roots=["input"])
        return found

    def input_file(self, project: Dict[str, Any], kind: str) -> Optional[Tuple[Dict[str, Any], Path]]:
        found = self.input_files(project, kind)
        return found[0] if found else None

    def copy_into_project(
        self,
        project: Dict[str, Any],
        source: Path,
        destination_directory: str,
        *,
        kind: str = "generated",
    ) -> Dict[str, Any]:
        project_dir = self.project_dir(project)
        destination_directory_path = resolve_within(project_dir, destination_directory)
        destination_directory_path.mkdir(parents=True, exist_ok=True)
        destination = unique_destination(destination_directory_path, Path(source).name)
        shutil.copy2(source, destination)
        return self.register_file(project, destination, kind=kind)
