"""Checkpointed, cancellable pipeline orchestration."""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Dict, Optional

from models import BUSSID_STAGES, LIGHT_STAGES, PHOTOGRAMMETRY_STAGES, STAGE_BY_KEY, STAGE_KEYS, STAGE_SPECS
from . import blender, bussid, exporter, filtering, frames, lights, meshroom, preprocessing, reconstruction, video
from pipeline.common import PipelineContext, StageError
from utils.subprocess import ProcessCancelled


class PipelineRunner:
    """Execute project stages in a background worker and persist checkpoints."""

    def __init__(self, project_manager: Any, db: Any, config: Any, settings: Any, logger: Any):
        self.project_manager = project_manager
        self.db = db
        self.config = config
        self.settings = settings
        self.logger = logger

    def _job_update(
        self,
        job_id: int,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        operation: Optional[str] = None,
        progress: Optional[float] = None,
        error: Any = "__KEEP__",
    ) -> None:
        fields = []
        values = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if stage is not None:
            fields.append("current_stage = ?")
            values.append(stage)
        if operation is not None:
            fields.append("operation = ?")
            values.append(operation)
        if progress is not None:
            fields.append("progress = ?")
            values.append(max(0.0, min(100.0, float(progress))))
        if error != "__KEEP__":
            fields.append("error = ?")
            if error is None:
                values.append(None)
            else:
                import json

                values.append(json.dumps(error, separators=(",", ":")))
        fields.append("updated_at = ?")
        values.extend((time.time(), job_id))
        self.db.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
            values,
        )

    def _emit(
        self,
        ctx: PipelineContext,
        *,
        key: str,
        operation: str,
        progress: float,
        status: str,
        log: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        event: Dict[str, Any] = {
            "type": "project_update",
            "project_id": ctx.project_id,
            "job_id": ctx.job_id,
            "stage": key,
            "stage_label": STAGE_BY_KEY[key]["label"],
            "operation": operation,
            "progress": round(max(0.0, min(100.0, progress)), 2),
            "status": status,
            "time": time.time(),
        }
        if log:
            event.update(
                {
                    "log": log.get("message"),
                    "level": log.get("level", "INFO"),
                    "operation": log.get("operation", operation),
                }
            )
        if error:
            event["error"] = error
        ctx.emit(event)

    def _has_video(self, ctx: PipelineContext) -> bool:
        return bool(ctx.input_file("video"))

    def _has_visual_input(self, ctx: PipelineContext) -> bool:
        return bool(ctx.input_file("video") or ctx.input_files("image"))

    def _has_model(self, ctx: PipelineContext) -> bool:
        return bool(ctx.model_file())

    def _should_skip(self, ctx: PipelineContext, key: str) -> bool:
        options = ctx.project.get("options") or {}
        has_visual_input = self._has_visual_input(ctx)
        if key in PHOTOGRAMMETRY_STAGES:
            if key == "preprocessing" and not bool(options.get("preprocessing", False)):
                return True
            return not bool(options.get("photogrammetry", True)) or not has_visual_input
        if key == "blender":
            return not bool(options.get("blender", True))
        if key in LIGHT_STAGES:
            return not bool(options.get("lights", False))
        if key == "bussid_prep":
            return not bool(options.get("bussid", False))
        if key in {"export", "package"}:
            # Generic stages: always run when a model exists.  BUSSID-only
            # projects keep working because their model is staged the same way.
            return not self._has_model(ctx)
        return False

    def _mark_skipped(self, ctx: PipelineContext, key: str, reason: str) -> None:
        progress = (STAGE_KEYS.index(key) / len(STAGE_KEYS)) * 100
        self.project_manager.set_stage(ctx.project_id, key, "skipped", result={"reason": reason})
        self.project_manager.update_runtime(
            ctx.project_id,
            status="running",
            current_stage=key,
            progress=progress,
        )
        self._job_update(
            ctx.job_id,
            status="running",
            stage=key,
            operation="skip",
            progress=progress,
        )
        log = ctx.log(
            reason,
            stage=STAGE_BY_KEY[key]["label"],
            operation="skip",
            status="skipped",
        )
        self._emit(
            ctx,
            key=key,
            operation="skip",
            progress=progress,
            status="running",
            log=log,
        )

    def run(
        self,
        project_id: int,
        job_id: int,
        control: Any,
        emit: Callable[[Dict[str, Any]], None],
    ) -> Dict[str, Any]:
        # The loop is intentionally generic.  Each stage module owns its
        # technical details, while this method owns ordering, checkpoints,
        # cancellation, progress, and event publication.
        project = self.project_manager.get(project_id)
        if not project:
            raise StageError("Project not found")
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
            emit=emit,
        )
        state = self.project_manager.state(project_id)
        for key, record in state.get("stages", {}).items():
            if key == "lights" and record.get("status") in {"done", "completed"}:
                ctx.results["light_analysis"] = record.get("result") or {}
        total = len(STAGE_KEYS)
        for index, key in enumerate(STAGE_KEYS):
            ctx.check()
            record = state.get("stages", {}).get(key, {})
            # Completed stages are skipped on resume.  Their output files and
            # the state record are the checkpoint; the worker does not replay
            # expensive video/Meshroom/Blender work unnecessarily.
            if record.get("status") in {"done", "completed"}:
                continue
            if self._should_skip(ctx, key):
                self._mark_skipped(ctx, key, "Stage is disabled or its required input is unavailable")
                continue

            label = STAGE_BY_KEY[key]["label"]
            before_progress = index / total * 100
            self.project_manager.set_stage(ctx.project_id, key, "running")
            self.project_manager.update_runtime(
                ctx.project_id,
                status="running",
                current_stage=key,
                progress=before_progress,
            )
            self._job_update(
                job_id,
                status="running",
                stage=key,
                operation="start",
                progress=before_progress,
            )
            start_log = ctx.log(
                f"{label} started",
                stage=label,
                operation="start",
                status="running",
            )
            self._emit(
                ctx,
                key=key,
                operation="start",
                progress=before_progress,
                status="running",
                log=start_log,
            )
            try:
                ctx.check()
                result = self._run_stage(ctx, key)
                ctx.check()
                after_progress = (index + 1) / total * 100
                self.project_manager.set_stage(
                    ctx.project_id,
                    key,
                    "done",
                    result=result if isinstance(result, dict) else {"value": result},
                )
                self.project_manager.update_runtime(
                    ctx.project_id,
                    status="running",
                    current_stage=key,
                    progress=after_progress,
                )
                self._job_update(
                    job_id,
                    status="running",
                    stage=key,
                    operation="complete",
                    progress=after_progress,
                )
                done_log = ctx.log(
                    f"{label} completed",
                    stage=label,
                    operation="complete",
                    status="completed",
                )
                self._emit(
                    ctx,
                    key=key,
                    operation="complete",
                    progress=after_progress,
                    status="running",
                    log=done_log,
                )
            except ProcessCancelled:
                self.project_manager.set_stage(ctx.project_id, key, "pending")
                raise
            except StageError as exc:
                error = {
                    "message": str(exc),
                    "stage": key,
                    "stage_label": label,
                    "exit_code": exc.exit_code,
                    "details": exc.details,
                }
                self.project_manager.set_stage(ctx.project_id, key, "failed", error=error)
                self.project_manager.update_runtime(
                    ctx.project_id,
                    status="failed",
                    current_stage=key,
                    error=error,
                )
                self._job_update(
                    job_id,
                    status="failed",
                    stage=key,
                    operation="failed",
                    progress=before_progress,
                    error=error,
                )
                failure_log = ctx.log(
                    str(exc),
                    stage=label,
                    operation="failed",
                    status="failed",
                    level="ERROR",
                    exit_code=exc.exit_code,
                )
                self._emit(
                    ctx,
                    key=key,
                    operation="failed",
                    progress=before_progress,
                    status="failed",
                    log=failure_log,
                    error=error,
                )
                raise
            except Exception as exc:
                error = {
                    "message": f"{label} failed: {exc}",
                    "stage": key,
                    "stage_label": label,
                    "details": {"traceback": traceback.format_exc(limit=8)},
                }
                self.project_manager.set_stage(ctx.project_id, key, "failed", error=error)
                self.project_manager.update_runtime(
                    ctx.project_id,
                    status="failed",
                    current_stage=key,
                    error=error,
                )
                self._job_update(
                    job_id,
                    status="failed",
                    stage=key,
                    operation="failed",
                    progress=before_progress,
                    error=error,
                )
                failure_log = ctx.log(
                    error["message"],
                    stage=label,
                    operation="failed",
                    status="failed",
                    level="ERROR",
                )
                self._emit(
                    ctx,
                    key=key,
                    operation="failed",
                    progress=before_progress,
                    status="failed",
                    log=failure_log,
                    error=error,
                )
                raise

        final_state = self.project_manager.state(project_id)
        final_state["last_result"] = {"completed_at": time.time()}
        self.project_manager.save_state(project_id, final_state)
        self.project_manager.update_runtime(
            project_id,
            status="completed",
            current_stage="completed",
            progress=100,
            error=None,
        )
        self._job_update(
            job_id,
            status="completed",
            stage="completed",
            operation="complete",
            progress=100,
        )
        final_log = ctx.log(
            "Project processing completed",
            stage="PIPELINE",
            operation="complete",
            status="completed",
        )
        self._emit(
            ctx,
            key="package",
            operation="complete",
            progress=100,
            status="completed",
            log=final_log,
        )
        return {"status": "completed", "progress": 100}

    def _run_stage(self, ctx: PipelineContext, key: str) -> Any:
        if key == "validation":
            if not self._has_video(ctx) and self._has_model(ctx):
                result = {"mode": "existing_model", "message": "Using uploaded 3D model"}
                ctx.log(
                    result["message"],
                    stage="VIDEO",
                    operation="validate",
                    status="completed",
                )
                return result
            return video.validate(ctx)
        if key == "extraction":
            return frames.extract(ctx)
        if key == "filtering":
            return frames.filter_frames(ctx)
        if key == "preprocessing":
            return frames.preprocess(ctx)
        if key == "meshroom":
            return meshroom.reconstruct(ctx)
        if key == "mesh_validation":
            return reconstruction.validate(ctx)
        if key == "blender":
            return blender.prepare(ctx)
        if key == "lights":
            return lights.analyze(ctx)
        if key == "light_creation":
            return lights.create(ctx)
        if key == "light_validation":
            return lights.validate(ctx)
        if key == "bussid_prep":
            return bussid.prepare(ctx)
        if key == "export":
            return exporter.export_model(ctx)
        if key == "package":
            return exporter.package_result(ctx)
        raise StageError(f"No implementation registered for stage {key}")
