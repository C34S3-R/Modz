"""Checkpointing, stage records, and memory monitoring.

Two promises this module keeps:

1. Section 24 - every major stage is restartable.  A shutdown during dense
   reconstruction must not force frame extraction to run again, so stage
   status lives on disk (``reconstruction/project_state.json``) and is
   written atomically after every transition.

2. Section 25 - every stage records start/end time, command, exit code,
   stdout, stderr, memory usage, output files, and a human-readable failure
   reason.  "Process failed." is not an acceptable error message.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .config import ReconstructionError
from utils.subprocess import ProcessCancelled


# Stage keys match the engine's STAGE_PIPELINE exactly: summary() drives the
# CLI and the status API, and a key that does not exist in the pipeline would
# show "pending" forever while the real record (written under the pipeline's
# key) stayed invisible.  Stages beyond the current phase exist as "pending"
# placeholders so the state file always shows the whole plan.
STAGE_ORDER: tuple[str, ...] = (
    "input_validation",
    "frame_extraction",
    "frame_selection",
    "feature_extraction",
    "feature_matching",
    "sfm",
    "masking",
    "dense",
    "mesh",
    "lowpoly",
    "texture",
    "blender",
    "bussid",
)

STAGE_LABELS: Dict[str, str] = {
    "input_validation": "Input Validation",
    "frame_extraction": "Frame Extraction",
    "frame_selection": "Frame Quality Analysis & Selection",
    "masking": "Background Masking (foreground isolation)",
    "feature_extraction": "SIFT Feature Detection & Description",
    "feature_matching": "Sequential Matching + Geometric Verification",
    "sfm": "Structure from Motion (sparse reconstruction)",
    "dense": "CPU Dense Reconstruction (plane-sweep depth maps)",
    "mesh": "Surface Reconstruction (TSDF fusion + marching cubes)",
    "lowpoly": "Mesh Decimation (low-poly)",
    "texture": "Texture Generation",
    "blender": "Blender Cleanup & Light Objects",
    "bussid": "BUSSID Preparation",
}

VALID_STATUSES = {"pending", "running", "complete", "failed", "skipped"}


def _now() -> float:
    return time.time()


def read_meminfo() -> Dict[str, float]:
    """System RAM snapshot in GB (Linux /proc/meminfo). Returns {} off-Linux."""
    out: Dict[str, float] = {}
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if not parts:
                    continue
                try:
                    out[key] = float(parts[0]) / (1024.0 * 1024.0)  # kB -> GB
                except ValueError:
                    continue
    except OSError:
        return {}
    return out


def process_rss_gb() -> float:
    """Resident set size of this process in GB (0.0 if unavailable)."""
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 ** 3)
    except (OSError, IndexError, ValueError):
        return 0.0


def peak_rss_gb() -> float:
    """Peak resident set size of this process in GB (VmHWM)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)  # kB -> GB
    except (OSError, IndexError, ValueError):
        pass
    return 0.0


def memory_pressure(limit_gb: float) -> Dict[str, Any]:
    """Are we close to the configured RAM ceiling?  Callers shrink batches."""
    info = read_meminfo()
    available = info.get("MemAvailable", 0.0)
    used_by_us = process_rss_gb()
    return {
        "limit_gb": limit_gb,
        "available_gb": round(available, 2),
        "process_rss_gb": round(used_by_us, 3),
        "process_peak_gb": round(peak_rss_gb(), 3),
        "over_limit": bool(used_by_us > limit_gb),
        "critical": bool(0 < available < 0.7),
    }


class StageContext:
    """One stage's record; filled in as it runs and persisted at the end."""

    def __init__(self, key: str, label: str, control: Any = None,
                 log_path: Optional[Path] = None) -> None:
        self.key = key
        self.label = label
        # Optional cancellation/pause control (the web backend's job control).
        # The standalone CLI leaves it None; both paths share this class.
        self.control = control
        self._log_path = log_path
        self.record: Dict[str, Any] = {
            "stage": key,
            "label": label,
            "status": "running",
            "start_time": _now(),
            "end_time": None,
            "elapsed_seconds": None,
            "command": None,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "memory_start_gb": round(process_rss_gb(), 3),
            "memory_peak_gb": None,
            "memory_system_available_gb": None,
            "output_files": [],
            "failure_reason": None,
            "error": None,
            "result": {},
        }

    def set_command(self, command: Any) -> None:
        self.record["command"] = command if isinstance(command, str) else " ".join(
            str(part) for part in command)

    def check(self) -> None:
        """Honour cancellation/pause between expensive units of work."""
        if self.control is not None:
            self.control.raise_if_cancelled()
            self.control.wait_if_paused()

    def set_output(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        # Cap stored output so a runaway ffmpeg cannot bloat the state file.
        self.record["stdout"] = (stdout or "")[-4000:]
        self.record["stderr"] = (stderr or "")[-4000:]
        self.record["exit_code"] = exit_code

    def add_output_files(self, paths: List[Path]) -> None:
        for path in paths:
            try:
                self.record["output_files"].append(str(path))
            except Exception:
                pass

    def result(self, value: Dict[str, Any]) -> None:
        """Store the stage's inspectable output (also returned to the caller)."""
        self.record["result"] = value

    def note(self, message: str) -> None:
        """Append a progress note visible in the stage record."""
        notes = self.record.setdefault("notes", [])
        notes.append(f"[{_now():.0f}] {message}")
        # Keep the record bounded but keep BOTH ends.  The first notes carry
        # the seed pair, the focal-probe scores and the calibration decisions
        # - exactly what is needed to diagnose a failed run - while the last
        # notes show how it ended.  Trimming only the head (the old
        # behaviour) discarded that diagnosis and left only the tail.
        if len(notes) > 240:
            del notes[60:-180:]

    def log(self, message: str, *, stage: Optional[str] = None,
            operation: Optional[str] = None, status: str = "info") -> None:
        """Append a human-readable line to the project pipeline log."""
        if self._log_path is None:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = (f"{stamp} [{status.upper()}] {stage or self.key} "
                f"{operation or '-'} | {message}\n")
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass

    def finish(self, status: str, result: Optional[Dict[str, Any]] = None,
               error: Optional[Dict[str, Any]] = None) -> None:
        end = _now()
        info = read_meminfo()
        self.record["end_time"] = end
        self.record["elapsed_seconds"] = round(end - self.record["start_time"], 2)
        self.record["status"] = status
        self.record["memory_peak_gb"] = round(peak_rss_gb(), 3)
        self.record["memory_system_available_gb"] = round(
            info.get("MemAvailable", 0.0), 2)
        if result:
            self.record["result"] = result
        if error:
            self.record["error"] = error
            self.record["failure_reason"] = error.get("message")


class StateStore:
    """Atomic JSON state + per-stage logs.  Safe to read while running."""

    def __init__(self, state_path: Path) -> None:
        self.path = Path(state_path)
        self.log_dir = self.path.parent.parent / "logs"
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {}
        self.load()

    # -- persistence -------------------------------------------------------
    def load(self) -> Dict[str, Any]:
        with self._lock:
            if self.path.is_file():
                try:
                    self._state = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    self._state = {}
            else:
                self._state = {}
            self._state.setdefault("stages", {})
            self._state.setdefault("created_at", _now())
            return self._state

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)  # atomic: a crash never half-writes

    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    # -- stage transitions ---------------------------------------------------
    def stage_record(self, key: str) -> Dict[str, Any]:
        return self._state["stages"].get(key, {})

    def status(self, key: str) -> str:
        return str(self.stage_record(key).get("status", "pending"))

    def is_complete(self, key: str) -> bool:
        return self.status(key) in {"complete", "completed", "done"}

    def begin(self, key: str, control: Any = None) -> StageContext:
        ctx = StageContext(key, STAGE_LABELS.get(key, key), control=control,
                           log_path=self.log_dir / "pipeline.log")
        with self._lock:
            self._state["stages"][key] = ctx.record
            self._state["updated_at"] = _now()
            self.save()
        return ctx

    def complete(self, ctx: StageContext, result: Dict[str, Any]) -> None:
        with self._lock:
            ctx.finish("complete", result=result)
            self._state["stages"][ctx.key] = ctx.record
            self._state["updated_at"] = _now()
            self.save()
            self._write_stage_log(ctx)

    def fail(self, ctx: StageContext, error: Dict[str, Any]) -> None:
        with self._lock:
            ctx.finish("failed", error=error)
            self._state["stages"][ctx.key] = ctx.record
            self._state["updated_at"] = _now()
            self.save()
            self._write_stage_log(ctx)

    def cancel(self, ctx: StageContext) -> None:
        """Cancellation/pause is not a failure: leave the stage resumable.

        The record returns to ``pending`` so the next run re-enters the
        stage from the top (whatever it was mid-way through did not
        finish), and no failure reason is recorded - a cancelled pipeline
        never "broke".  The web runner maps ``ProcessCancelled`` back to
        its own cancelled semantics; the CLI simply resumes later.
        """
        with self._lock:
            ctx.finish("pending", result={"reason": "cancelled"})
            ctx.record["failure_reason"] = None
            self._state["stages"][ctx.key] = ctx.record
            self._state["updated_at"] = _now()
            self.save()
            self._write_stage_log(ctx)

    def mark_skipped(self, key: str, reason: str) -> None:
        with self._lock:
            record = self.stage_record(key)
            record.update({
                "stage": key,
                "label": STAGE_LABELS.get(key, key),
                "status": "skipped",
                "failure_reason": None,
                "result": {"reason": reason},
                "end_time": _now(),
            })
            self._state["stages"][key] = record
            self.save()

    def _write_stage_log(self, ctx: StageContext) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"stage_{ctx.key}.json"
            path.write_text(
                json.dumps(ctx.record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    # -- reporting ------------------------------------------------------------
    def summary(self) -> List[Dict[str, Any]]:
        rows = []
        for key in STAGE_ORDER:
            rec = self.stage_record(key)
            rows.append({
                "stage": key,
                "label": STAGE_LABELS.get(key, key),
                "status": rec.get("status", "pending"),
                "elapsed_seconds": rec.get("elapsed_seconds"),
                "failure_reason": rec.get("failure_reason"),
            })
        return rows

    @contextmanager
    def stage(self, key: str, control: Any = None) -> Iterator[StageContext]:
        """Run a stage with full section-25 record keeping.

        On success the record (outputs, timings, memory) is persisted.  On
        failure a structured error with a human-readable reason is persisted
        and a ReconstructionError re-raised - the caller decides how to
        surface it, but the state file always explains what broke.
        """
        ctx = self.begin(key, control=control)
        try:
            yield ctx
        except ProcessCancelled:
            # Cancel/pause (the web JobControl raises this, the CLI can
            # too) must not masquerade as a stage failure: reset the
            # record to pending so resume re-enters it, then let the
            # caller keep its own cancelled semantics.
            self.cancel(ctx)
            raise
        except ReconstructionError as exc:
            error = {
                "message": exc.human(),
                "reason": exc.message,
                "suggestion": exc.suggestion,
                "details": exc.details,
                "exit_code": exc.exit_code,
                "traceback": traceback.format_exc(limit=6),
            }
            self.fail(ctx, error)
            raise
        except Exception as exc:  # noqa: BLE001 - stage boundary
            error = {
                "message": f"{STAGE_LABELS.get(key, key)} failed: {exc}",
                "reason": str(exc),
                "suggestion": "Check logs/stage_%s.json and logs/pipeline.log "
                              "for the command and output that led here." % key,
                "details": {"traceback": traceback.format_exc(limit=8)},
            }
            self.fail(ctx, error)
            raise ReconstructionError(
                error["message"],
                suggestion=error["suggestion"],
                details={"stage": key},
            ) from exc
        else:
            self.complete(ctx, ctx.record.get("result") or {})
