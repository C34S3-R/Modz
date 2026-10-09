"""Single-worker job queue with pause, cancellation, and restart recovery."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from fastapi import HTTPException
from models import public_job
from pipeline.common import StageError
from utils.subprocess import ProcessCancelled


class JobControl:
    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()
        self.pause_event.clear()

    def request_pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise ProcessCancelled("Processing cancelled")

    def wait_if_paused(self) -> None:
        while self.pause_event.wait(timeout=0.2):
            self.raise_if_cancelled()


class ResourceGate:
    """Limit heavy work while keeping the worker implementation simple.

    Meshroom and Blender can consume most of the server memory, so this gate
    is shared by pipeline jobs and export jobs rather than being local to one
    worker function.
    """

    def __init__(self, maximum: int):
        self._condition = threading.Condition()
        self._maximum = max(1, maximum)
        self._active = 0

    def set_maximum(self, maximum: int) -> None:
        with self._condition:
            self._maximum = max(1, maximum)
            self._condition.notify_all()

    @property
    def maximum(self) -> int:
        with self._condition:
            return self._maximum

    @contextmanager
    def acquire(self) -> Iterator[None]:
        with self._condition:
            while self._active >= self._maximum:
                self._condition.wait(timeout=1.0)
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()


class JobManager:
    def __init__(
        self,
        db: Any,
        project_manager: Any,
        pipeline_runner: Any,
        config: Any,
        logger: Any,
        broker: Any,
    ):
        self.db = db
        self.project_manager = project_manager
        self.pipeline_runner = pipeline_runner
        self.config = config
        self.logger = logger
        self.broker = broker
        self.resource_gate = ResourceGate(config.max_concurrent_jobs)
        self._condition = threading.Condition()
        self._stopping = False
        self._threads: list[threading.Thread] = []
        self._controls: Dict[int, JobControl] = {}

    def _spawn_worker(self, index: int) -> None:
        thread = threading.Thread(
            target=self._worker_loop,
            name=f"vehicle-job-worker-{index + 1}",
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    def start(self) -> None:
        self._recover_incomplete()
        if any(thread.is_alive() for thread in self._threads):
            return
        self._stopping = False
        self._threads = []
        for index in range(self.config.max_concurrent_jobs):
            self._spawn_worker(index)

    def reconfigure(self) -> None:
        """Apply a settings change to the shared gate and add workers if needed."""

        self.resource_gate.set_maximum(self.config.max_concurrent_jobs)
        alive = sum(thread.is_alive() for thread in self._threads)
        for index in range(alive, self.config.max_concurrent_jobs):
            self._spawn_worker(index)

    def stop(self) -> None:
        self._stopping = True
        for control in list(self._controls.values()):
            control.request_cancel()
        with self._condition:
            self._condition.notify_all()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=10)

    def _recover_incomplete(self) -> None:
        rows = self.db.fetchall(
            "SELECT * FROM jobs WHERE type = 'pipeline' AND status IN ('running','paused','cancelling')"
        )
        for row in rows:
            error = {
                "message": "Server restarted while the job was active",
                "stage": row.get("current_stage"),
            }
            self._set_job(
                row["id"],
                status="failed",
                finished_at=time.time(),
                error=error,
            )
            self.project_manager.update_runtime(
                row["project_id"],
                status="failed",
                error=error,
            )
        # Queued jobs remain durable in SQLite and are picked up again.
        for row in self.db.fetchall(
            "SELECT project_id FROM jobs WHERE type = 'pipeline' AND status = 'queued'"
        ):
            self.project_manager.update_runtime(
                row["project_id"],
                status="queued",
                current_stage="queued",
            )

    def _set_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields.setdefault("updated_at", time.time())
        columns = []
        values = []
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

    def _event(
        self,
        project_id: int,
        *,
        status: str,
        stage: Optional[str] = None,
        operation: Optional[str] = None,
        progress: Optional[float] = None,
        job_id: Optional[int] = None,
        error: Any = None,
    ) -> None:
        event = {
            "type": "job_update",
            "project_id": project_id,
            "job_id": job_id,
            "status": status,
            "stage": stage,
            "operation": operation,
            "progress": progress,
            "time": time.time(),
        }
        if error is not None:
            event["error"] = error
        self.broker.publish(event)

    def active_job(
        self,
        project_id: int,
        job_type: str = "pipeline",
    ) -> Optional[Dict[str, Any]]:
        return self.db.fetchone(
            """
            SELECT * FROM jobs
            WHERE project_id = ? AND type = ?
              AND status IN ('queued','running','paused','cancelling')
            ORDER BY id DESC LIMIT 1
            """,
            (project_id, job_type),
        )

    def enqueue_pipeline(self, project_id: int, *, retry: bool = False) -> Dict[str, Any]:
        if self.project_manager.is_deleting(project_id):
            raise HTTPException(status_code=409, detail="Project deletion is in progress")
        project = self.project_manager.get_or_404(project_id)
        active = self.active_job(project_id)
        if active:
            if active["status"] == "paused":
                self.resume(active["id"])
            return self.get_job(active["id"])  # type: ignore[return-value]
        if retry:
            self.project_manager.reset_for_retry(project_id)
        timestamp = time.time()
        try:
            job_id = self.db.execute(
                """
                INSERT INTO jobs
                    (project_id, type, status, current_stage, operation, progress,
                     created_at, updated_at)
                VALUES (?, 'pipeline', 'queued', NULL, 'queued', 0, ?, ?)
                """,
                (project_id, timestamp, timestamp),
            )
        except sqlite3.IntegrityError:
            active = self.active_job(project_id)
            if active:
                return self.get_job(active["id"])  # type: ignore[return-value]
            raise
        self.project_manager.update_runtime(
            project_id,
            status="queued",
            current_stage="queued",
            progress=self.project_manager._completed_progress(
                self.project_manager.state(project_id)
            ),
            error=None,
        )
        self.logger.append(
            project_id,
            self.project_manager.project_dir(project),
            stage="JOB",
            operation="queue",
            status="queued",
            message=f"Queued processing job {job_id}",
        )
        self._event(
            project_id,
            job_id=job_id,
            status="queued",
            stage="queued",
            operation="queue",
            progress=0,
        )
        with self._condition:
            self._condition.notify_all()
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        row = self.db.fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return public_job(row) if row else None

    def get_job_or_404(self, job_id: int) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    def list_jobs(self, project_id: Optional[int] = None) -> list[Dict[str, Any]]:
        if project_id is None:
            rows = self.db.fetchall("SELECT * FROM jobs ORDER BY id DESC")
        else:
            rows = self.db.fetchall(
                "SELECT * FROM jobs WHERE project_id = ? ORDER BY id DESC",
                (project_id,),
            )
        return [public_job(row) for row in rows]

    def _resources_available(self) -> tuple[bool, str]:
        try:
            if shutil.disk_usage(self.config.base_dir).free < 512 * 1024 * 1024:
                return False, "less than 512 MB of disk space is available"
        except OSError:
            pass
        try:
            values = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, raw = line.split(":", 1)
                values[key] = float(raw.strip().split()[0]) * 1024
            if values.get("MemAvailable", 0) < 512 * 1024 * 1024:
                return False, "less than 512 MB of available memory"
        except (OSError, ValueError, IndexError):
            pass
        return True, ""

    def _worker_loop(self) -> None:
        # Workers poll SQLite rather than keeping an in-memory queue.  This
        # lets a queued job survive a process restart and keeps HTTP handlers
        # from owning processing work.
        while True:
            with self._condition:
                while not self._stopping:
                    row = self.db.fetchone(
                        "SELECT * FROM jobs WHERE type = 'pipeline' AND status = 'queued' ORDER BY id ASC LIMIT 1"
                    )
                    if row:
                        break
                    self._condition.wait(timeout=1.0)
                if self._stopping:
                    return
                job = self.db.fetchone("SELECT * FROM jobs WHERE id = ?", (row["id"],))
            if not job:
                continue
            available, reason = self._resources_available()
            if not available:
                self._set_job(
                    job["id"],
                    status="queued",
                    current_stage="waiting_resources",
                    operation="wait",
                )
                self.project_manager.update_runtime(
                    job["project_id"],
                    status="queued",
                    current_stage="waiting_resources",
                )
                self._event(
                    job["project_id"],
                    job_id=job["id"],
                    status="queued",
                    stage="waiting_resources",
                    operation="wait",
                )
                time.sleep(2)
                continue
            control = JobControl()
            self._controls[job["id"]] = control
            # Claim the row conditionally.  Two workers may observe the same
            # queued row; only one UPDATE can change it from queued to running.
            claimed = self.db.execute(
                """
                UPDATE jobs SET status = 'running', started_at = ?, current_stage = ?,
                    operation = ?, updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (job.get("started_at") or time.time(), "starting", "start", time.time(), job["id"]),
            )
            if not claimed:
                self._controls.pop(job["id"], None)
                continue
            self.project_manager.update_runtime(
                job["project_id"],
                status="running",
                current_stage="starting",
            )
            self.logger.append(
                job["project_id"],
                self.project_manager.project_dir(self.project_manager.get_or_404(job["project_id"])),
                stage="JOB",
                operation="start",
                status="running",
                message=f"Started processing job {job['id']}",
            )
            self._event(
                job["project_id"],
                job_id=job["id"],
                status="running",
                stage="starting",
                operation="start",
            )
            try:
                with self.resource_gate.acquire():
                    self.pipeline_runner.run(
                        job["project_id"],
                        job["id"],
                        control,
                        self.broker.publish,
                    )
            except ProcessCancelled:
                self._finish_cancelled(job)
            except StageError as exc:
                # The runner normally persists the detailed stage error.  Keep a
                # fallback here for errors raised before a stage starts.
                self._finish_failed(job, {"message": str(exc), "stage": job.get("current_stage")})
            except Exception as exc:
                self._finish_failed(
                    job,
                    {
                        "message": f"Processing failed: {exc}",
                        "stage": job.get("current_stage"),
                    },
                )
            else:
                self._finish_completed(job)
            finally:
                self._controls.pop(job["id"], None)
                with self._condition:
                    self._condition.notify_all()

    def _finish_completed(self, job: Dict[str, Any]) -> None:
        finished = time.time()
        self._set_job(
            job["id"],
            status="completed",
            current_stage="completed",
            operation="complete",
            progress=100,
            finished_at=finished,
            error=None,
        )
        self.project_manager.update_runtime(
            job["project_id"],
            status="completed",
            current_stage="completed",
            progress=100,
            error=None,
        )
        self.logger.append(
            job["project_id"],
            self.project_manager.project_dir(self.project_manager.get_or_404(job["project_id"])),
            stage="JOB",
            operation="complete",
            status="completed",
            message=f"Processing job {job['id']} completed",
        )
        self._event(
            job["project_id"],
            job_id=job["id"],
            status="completed",
            stage="completed",
            operation="complete",
            progress=100,
        )

    def _finish_failed(self, job: Dict[str, Any], error: Dict[str, Any]) -> None:
        current = self.get_job(job["id"]) or job
        if current.get("status") != "failed":
            self._set_job(
                job["id"],
                status="failed",
                finished_at=time.time(),
                error=error,
            )
            self.project_manager.update_runtime(
                job["project_id"],
                status="failed",
                error=error,
            )
            self.logger.append(
                job["project_id"],
                self.project_manager.project_dir(self.project_manager.get_or_404(job["project_id"])),
                stage="JOB",
                operation="failed",
                status="failed",
                message=error.get("message", "Processing failed"),
                level="ERROR",
            )
        elif current.get("finished_at") is None:
            self._set_job(job["id"], finished_at=time.time())
        self._event(
            job["project_id"],
            job_id=job["id"],
            status="failed",
            stage=error.get("stage") or current.get("current_stage"),
            operation="failed",
            error=error,
            progress=current.get("progress"),
        )

    def _finish_cancelled(self, job: Dict[str, Any]) -> None:
        error = {"message": "Processing cancelled by user", "cancelled": True}
        self._set_job(
            job["id"],
            status="cancelled",
            current_stage="cancelled",
            operation="cancelled",
            finished_at=time.time(),
            error=error,
        )
        self.project_manager.update_runtime(
            job["project_id"],
            status="cancelled",
            current_stage="cancelled",
            error=error,
        )
        self.logger.append(
            job["project_id"],
            self.project_manager.project_dir(self.project_manager.get_or_404(job["project_id"])),
            stage="JOB",
            operation="cancel",
            status="cancelled",
            message="Processing cancelled by user",
            level="WARNING",
        )
        self._event(
            job["project_id"],
            job_id=job["id"],
            status="cancelled",
            stage="cancelled",
            operation="cancel",
        )

    def pause(self, job_id: int) -> Dict[str, Any]:
        job = self.get_job_or_404(job_id)
        if job["status"] != "running":
            raise HTTPException(status_code=409, detail="Only a running job can be paused")
        control = self._controls.get(job_id)
        if not control:
            raise HTTPException(status_code=409, detail="Job is not currently in the worker")
        control.request_pause()
        self._set_job(job_id, status="paused", operation="pause")
        self.project_manager.update_runtime(job["project_id"], status="paused")
        self._event(
            job["project_id"],
            job_id=job_id,
            status="paused",
            stage=job.get("current_stage"),
            operation="pause",
            progress=job.get("progress"),
        )
        return self.get_job_or_404(job_id)

    def resume(self, job_id: int) -> Dict[str, Any]:
        job = self.get_job_or_404(job_id)
        control = self._controls.get(job_id)
        if job["status"] == "paused" and control:
            control.resume()
            self._set_job(job_id, status="running", operation="resume")
            self.project_manager.update_runtime(job["project_id"], status="running")
            self._event(
                job["project_id"],
                job_id=job_id,
                status="running",
                stage=job.get("current_stage"),
                operation="resume",
                progress=job.get("progress"),
            )
            return self.get_job_or_404(job_id)
        if job["status"] == "queued":
            with self._condition:
                self._condition.notify_all()
            return job
        raise HTTPException(status_code=409, detail="Only a paused or queued job can be resumed")

    def cancel(self, job_id: int) -> Dict[str, Any]:
        job = self.get_job_or_404(job_id)
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        control = self._controls.get(job_id)
        if job["status"] == "queued" and not control:
            self._finish_cancelled(job)
            return self.get_job_or_404(job_id)
        if not control:
            raise HTTPException(status_code=409, detail="Job is not currently in the worker")
        control.request_cancel()
        self._set_job(job_id, status="cancelling", operation="cancel")
        self.project_manager.update_runtime(job["project_id"], status="cancelling")
        self._event(
            job["project_id"],
            job_id=job_id,
            status="cancelling",
            stage=job.get("current_stage"),
            operation="cancel",
            progress=job.get("progress"),
        )
        return self.get_job_or_404(job_id)

    def retry(self, project_id: int) -> Dict[str, Any]:
        project = self.project_manager.get_or_404(project_id)
        active = self.active_job(project_id)
        if active:
            raise HTTPException(status_code=409, detail="Project already has an active job")
        return self.enqueue_pipeline(project_id, retry=True)
