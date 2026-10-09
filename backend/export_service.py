"""Asynchronous BUSSID export/package service."""

from __future__ import annotations

import json
import sqlite3
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Optional

from fastapi import HTTPException

from job_manager import JobControl, ResourceGate
from models import public_job
from pipeline.common import PipelineContext
from pipeline import exporter
from utils.logger import ProjectLogger
from utils.subprocess import ProcessCancelled


class ExportService:
    def __init__(
        self,
        db: Any,
        project_manager: Any,
        config: Any,
        settings: Any,
        logger: ProjectLogger,
        broker: Any,
        resource_gate: ResourceGate,
    ):
        self.db = db
        self.project_manager = project_manager
        self.config = config
        self.settings = settings
        self.logger = logger
        self.broker = broker
        self.resource_gate = resource_gate
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vehicle-export")
        self._futures: Dict[int, Future] = {}
        self._controls: Dict[int, JobControl] = {}
        self._recover_incomplete()

    def _recover_incomplete(self) -> None:
        rows = self.db.fetchall(
            "SELECT * FROM jobs WHERE type = 'export' AND status IN ('queued','running','paused','cancelling')"
        )
        for row in rows:
            error = {
                "message": "Server restarted before the export could finish",
            }
            self._set_job(
                row["id"],
                status="failed",
                finished_at=time.time(),
                error=error,
            )
            state = self.project_manager.state(row["project_id"])
            if (state.get("export") or {}).get("job_id") == row["id"]:
                state["export"].update({"status": "failed", "error": error})
                self.project_manager.save_state(row["project_id"], state)

    def close(self) -> None:
        for control in list(self._controls.values()):
            control.request_cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _event(self, project_id: int, job_id: int, status: str, progress: float, **extra: Any) -> None:
        event = {
            "type": "export_update",
            "project_id": project_id,
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "time": time.time(),
        }
        event.update(extra)
        self.broker.publish(event)

    def _set_job(self, job_id: int, **fields: Any) -> None:
        fields.setdefault("updated_at", time.time())
        values = []
        columns = []
        for key, value in fields.items():
            columns.append(f"{key} = ?")
            if key == "error" and value is not None and not isinstance(value, str):
                value = json.dumps(value, separators=(",", ":"))
            values.append(value)
        values.append(job_id)
        self.db.execute(
            f"UPDATE jobs SET {', '.join(columns)} WHERE id = ?",
            values,
        )

    def status(self, project_id: int) -> Dict[str, Any]:
        project = self.project_manager.get_or_404(project_id)
        state = self.project_manager.state(project_id)
        export_state = dict(state.get("export") or {"status": "idle", "progress": 0})
        job_id = export_state.get("job_id")
        job = self.db.fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,)) if job_id else None
        if job:
            export_state["job"] = public_job(job)
        export_state.setdefault("status", "idle")
        export_state.setdefault("progress", 0)
        export_state["download_url"] = export_state.get("download_url")
        export_state["filename"] = export_state.get("filename")
        export_state["error"] = export_state.get("error")
        return export_state

    def checklist(self, project_id: int) -> list[Dict[str, Any]]:
        project = self.project_manager.get_or_404(project_id)
        model = self.project_manager.file_manager.model_file(project)
        options = project.get("options") or {}
        bussid_enabled = bool(options.get("bussid", False))
        staging = self.project_manager.project_dir(project) / "bussid"
        checks = [
            {"name": "Project exists", "ok": True},
            {"name": "3D model available", "ok": bool(model)},
            {
                "name": "BUSSID staging directory available" if bussid_enabled else "Export staging directory available",
                "ok": staging.is_dir() or (self.project_manager.project_dir(project) / "exports").is_dir(),
            },
            {"name": "Processing completed", "ok": project.get("status") in {"completed", "done"}},
        ]
        if not model:
            checks.append({"name": "Upload a model before export", "ok": False})
        return checks

    def start(self, project_id: int) -> Dict[str, Any]:
        if self.project_manager.is_deleting(project_id):
            raise HTTPException(status_code=409, detail="Project deletion is in progress")
        project = self.project_manager.get_or_404(project_id)
        state = self.project_manager.state(project_id)
        current = state.get("export") or {}
        if current.get("status") in {"queued", "running"}:
            return self.status(project_id)
        if self.db.fetchone(
            """
            SELECT id FROM jobs
            WHERE project_id = ? AND type = 'pipeline'
              AND status IN ('queued','running','paused','cancelling')
            LIMIT 1
            """,
            (project_id,),
        ):
            raise HTTPException(
                status_code=409,
                detail="Wait for processing to finish before starting an export",
            )
        if not self.project_manager.file_manager.model_file(project):
            raise HTTPException(status_code=400, detail="A 3D model is required before export")
        timestamp = time.time()
        try:
            job_id = self.db.execute(
                """
                INSERT INTO jobs
                    (project_id, type, status, current_stage, operation, progress, created_at, updated_at)
                VALUES (?, 'export', 'queued', 'export', 'queued', 0, ?, ?)
                """,
                (project_id, timestamp, timestamp),
            )
        except sqlite3.IntegrityError:
            active = self.db.fetchone(
                """
                SELECT * FROM jobs
                WHERE project_id = ? AND type = 'export'
                  AND status IN ('queued','running','paused','cancelling')
                ORDER BY id DESC LIMIT 1
                """,
                (project_id,),
            )
            if active:
                return self.status(project_id)
            raise
        state["export"] = {
            "status": "queued",
            "progress": 0,
            "job_id": job_id,
            "filename": None,
            "download_url": None,
            "error": None,
        }
        self.project_manager.save_state(project_id, state)
        self.logger.append(
            project_id,
            self.project_manager.project_dir(project),
            stage="EXPORT",
            operation="queue",
            status="queued",
            message=f"Queued export job {job_id}",
        )
        self._event(project_id, job_id, "queued", 0, operation="queue")
        self._futures[job_id] = self.executor.submit(self._run, project_id, job_id)
        return self.status(project_id)

    def cancel(self, project_id: int) -> Dict[str, Any]:
        self.project_manager.get_or_404(project_id)
        state = self.project_manager.state(project_id)
        export_state = state.get("export") or {}
        job_id = export_state.get("job_id")
        if not job_id:
            raise HTTPException(status_code=404, detail="No export job to cancel")
        job = self.db.fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not job or job["status"] in {"completed", "failed", "cancelled"}:
            return self.status(project_id)
        control = self._controls.get(job_id)
        future = self._futures.get(job_id)
        if future is not None and future.cancel():
            self._set_job(job_id, status="cancelled", operation="cancelled", finished_at=time.time())
            export_state.update(
                {
                    "status": "cancelled",
                    "error": {"message": "Export cancelled"},
                    "finished_at": time.time(),
                }
            )
            self.project_manager.save_state(project_id, state)
            self._event(project_id, job_id, "cancelled", 0, operation="cancel")
            return self.status(project_id)
        if control is None:
            raise HTTPException(status_code=409, detail="Export is not currently running")
        control.request_cancel()
        self._set_job(job_id, status="cancelling", operation="cancel")
        export_state.update({"status": "cancelling"})
        self.project_manager.save_state(project_id, state)
        self._event(project_id, job_id, "cancelling", export_state.get("progress", 0), operation="cancel")
        return self.status(project_id)

    def _run(self, project_id: int, job_id: int) -> None:
        control = JobControl()
        self._controls[job_id] = control
        self._set_job(job_id, status="running", started_at=time.time(), operation="start", progress=0)
        project = self.project_manager.get_or_404(project_id)
        state = self.project_manager.state(project_id)
        state["export"].update({"status": "running", "progress": 0, "job_id": job_id})
        self.project_manager.save_state(project_id, state)
        self._event(project_id, job_id, "running", 0, operation="start")
        ctx = PipelineContext(
            project=project,
            project_manager=self.project_manager,
            db=self.db,
            config=self.config,
            settings=self.settings,
            logger=self.logger,
            file_manager=self.project_manager.file_manager,
            control=control,
            job_id=job_id,
            emit=self.broker.publish,
        )
        try:
            with self.resource_gate.acquire():
                result = exporter.package_result(ctx)
            state = self.project_manager.state(project_id)
            state["export"].update(
                {
                    "status": "completed",
                    "progress": 100,
                    "filename": result.get("filename"),
                    "download_url": result.get("download_url"),
                    "error": None,
                    "finished_at": time.time(),
                    "asset_count": result.get("asset_count", 0),
                    "size": result.get("size", 0),
                    "path": result.get("path"),
                    "warnings": result.get("warnings", []),
                }
            )
            self.project_manager.save_state(project_id, state)
            self._set_job(
                job_id,
                status="completed",
                current_stage="completed",
                operation="complete",
                progress=100,
                finished_at=time.time(),
                error=None,
            )
            self._event(
                project_id,
                job_id,
                "completed",
                100,
                operation="complete",
                filename=result.get("filename"),
                download_url=result.get("download_url"),
            )
        except ProcessCancelled:
            error = {"message": "Export cancelled by user", "cancelled": True}
            state = self.project_manager.state(project_id)
            state["export"].update(
                {
                    "status": "cancelled",
                    "error": error,
                    "finished_at": time.time(),
                }
            )
            self.project_manager.save_state(project_id, state)
            self._set_job(
                job_id,
                status="cancelled",
                current_stage="cancelled",
                operation="cancelled",
                finished_at=time.time(),
                error=error,
            )
            self._event(project_id, job_id, "cancelled", 0, operation="cancel", error=error)
        except Exception as exc:
            error = {"message": str(exc)}
            state = self.project_manager.state(project_id)
            state["export"].update(
                {
                    "status": "failed",
                    "progress": state.get("export", {}).get("progress", 0),
                    "error": error,
                    "finished_at": time.time(),
                }
            )
            self.project_manager.save_state(project_id, state)
            self._set_job(
                job_id,
                status="failed",
                current_stage="export",
                operation="failed",
                finished_at=time.time(),
                error=error,
            )
            self.logger.append(
                project_id,
                self.project_manager.project_dir(project),
                stage="EXPORT",
                operation="failed",
                status="failed",
                message=str(exc),
                level="ERROR",
            )
            self._event(project_id, job_id, "failed", 0, operation="failed", error=error)
        finally:
            self._controls.pop(job_id, None)
            self._futures.pop(job_id, None)
