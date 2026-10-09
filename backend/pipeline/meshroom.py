"""Meshroom process boundary, bridged to the CPU-only reconstruction engine.

Historically this stage shelled out to an external Meshroom/AliceVision
build.  Its dense reconstruction (DepthMap) is CUDA-only, so on a host
without NVIDIA every run died at Meshroom stage 5; the project replaced
that path (option 3) with ``backend/reconstruction`` - a checkpointed,
CPU-only pipeline that goes video/photos -> sparse -> dense -> mesh ->
low-poly -> textured OBJ.  This module is the Phase 10 hook: the web
pipeline keeps its stage sequence, checkpoints, job control and live
events, but the stage named ``meshroom`` now runs the engine.

Contract preserved for everything downstream:

- runs inside the single job worker (never in a request handler), one
  heavy stage at a time;
- honours pause/cancel through ``ctx.control`` - the engine treats
  ``ProcessCancelled`` as "not a failure" and leaves its stage resumable;
- mirrors engine sub-stage transitions onto the project log so the live
  UI shows which internal stage is working;
- hands mesh validation / blender / lights / bussid / export a
  self-contained model registered under ``reconstruction/`` - the root
  ``mesh_validation`` scans first and ``model_file()`` prefers after
  ``blender/`` and ``bussid/``.  Names are kept verbatim so the OBJ's
  ``mtllib`` line and the MTL's ``map_Kd`` resolve next to the copies.

The old external path is kept, opt-in with ``MESHROOM_BACKEND=external``
for hosts that do have a working Meshroom install.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict

from .common import PipelineContext, StageError
from utils.subprocess import CommandError, run_command


def reconstruct(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.check()
    if os.getenv("MESHROOM_BACKEND", "").strip().lower() == "external":
        return _reconstruct_external(ctx)
    return _reconstruct_engine(ctx)


# --------------------------------------------------------------------------
def _reconstruct_engine(ctx: PipelineContext) -> Dict[str, Any]:
    """Run the CPU-only engine through the texture stage.

    Everything after geometry/texture stays with the web's own stages
    (blender, lights, bussid_prep, export, package): they own the DB
    registrations and download URLs the frontend consumes, and the engine
    deliberately does not duplicate that work for the web path.  For the
    standalone CLI (``vehicle-reconstruct``) the same engine runs its own
    blender/bussid stages - see reconstruction/README.md.
    """
    from reconstruction import ReconConfig, run_pipeline
    from reconstruction.config import ReconstructionError
    from reconstruction.state import STAGE_LABELS

    # The web UI owns fps / max_frames per project; the engine owns the
    # frame interval.  Bridge them so "more frames" in project options
    # actually reaches the dense sweep instead of staying at the preset.
    options = ctx.project.get("options", {}) or {}
    try:
        fps = float(options.get("fps", 4))
    except (TypeError, ValueError):
        fps = 4.0
    fps = max(1.0, min(60.0, fps))
    try:
        options_max_frames = int(options.get("max_frames", 300))
    except (TypeError, ValueError):
        options_max_frames = 300
    options_max_frames = max(1, min(10000, options_max_frames))
    # Project presets choose engine effort too: a chess piece must not pay
    # vehicle-grade dense reconstruction.  Explicit RECON_QUALITY env still
    # wins when set (operator override).
    preset = str(ctx.project.get("preset") or "standard").strip().lower()
    engine_quality = {"small": "low", "standard": "medium", "vehicle": "medium"}.get(preset, "medium")
    mask_background = options.get("mask_background")
    if mask_background is None:
        # Projects created before this option existed: mask general
        # subjects, leave vehicle street scenes to the guard.
        mask_background = (str(ctx.project.get("project_type") or "general") != "vehicle")
    mask_background = bool(mask_background)
    cfg = ReconConfig.for_project(
        ctx.project_dir,
        quality=os.getenv("RECON_QUALITY") or engine_quality,
        frame_interval=1.0 / fps,
        mask_background=mask_background,
    )
    # Never use fewer frames than the project asked for: the preset is a
    # floor for quality, the project option is the operator's dense request.
    cfg.max_frames = max(cfg.max_frames, options_max_frames)

    def _on_progress(event: Dict[str, Any]) -> None:
        key = str(event.get("stage") or "")
        label = STAGE_LABELS.get(key, key)
        status = str(event.get("status") or "")
        if status == "running":
            ctx.log(f"Engine stage starting: {label}", stage="MESHROOM",
                    operation=key, status="running")
        elif status == "complete":
            ctx.log(f"Engine stage complete: {label}", stage="MESHROOM",
                    operation=key, status="completed")
        elif status == "skipped":
            ctx.log(f"Engine stage reused from checkpoint: {label}",
                    stage="MESHROOM", operation=key, status="skipped")
        elif status == "failed":
            ctx.log(f"Engine stage failed: {label} - {event.get('error', '')}",
                    stage="MESHROOM", operation=key, status="failed",
                    level="ERROR")

    try:
        summary = run_pipeline(cfg, control=ctx.control,
                               on_progress=_on_progress, through="texture")
    except ReconstructionError as exc:
        raise StageError(
            f"MESHROOM FAILED: {exc.human()}",
            exit_code=exc.exit_code,
            details={"suggestion": exc.suggestion,
                     "engine": exc.details,
                     "checkpoint": "reconstruction/project_state.json"},
        ) from exc

    # Bridge the result into the shape the web stages already expect: a
    # self-contained model under reconstruction/.  Copy the sidecars under
    # their original names so mtllib/map_Kd resolve next to the OBJ -
    # mesh_validation fails the stage if the material reference dangles.
    texture_dir = cfg.d("texture")
    model = texture_dir / "textured.obj"
    if not model.is_file():
        raise StageError(
            "MESHROOM FAILED: the engine finished without producing "
            "texture/textured.obj",
            exit_code=1,
            details={"checkpoint": "reconstruction/project_state.json"},
        )
    output_dir = ctx.project_dir / "reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)
    bridged: list[str] = []
    for name, kind in (("textured.obj", "model"),
                       ("textured.mtl", "metadata"),
                       ("texture.png", "texture")):
        source = texture_dir / name
        if not source.is_file():
            raise StageError(
                f"MESHROOM FAILED: the engine output is missing {name}",
                exit_code=1,
                details={"expected": str(source)},
            )
        destination = output_dir / name
        shutil.copy2(source, destination)
        ctx.register(destination, kind)
        bridged.append(f"reconstruction/{name}")

    # Extra artifacts for the file browser (downloads).  Registered under
    # their own roots, which model_file() never scans, so they cannot
    # shadow the bridged model.
    for rel, kind in (("mesh/raw/mesh.ply", "model"),
                      ("lowpoly/lowpoly.ply", "model"),
                      ("texture/textured.obj", "model"),
                      ("texture/texture.png", "texture"),
                      ("dense/dense_point_cloud.ply", "model")):
        artifact = ctx.project_dir / rel
        if artifact.is_file():
            ctx.register(artifact, kind)
    report = ctx.project_dir / "reconstruction_report.json"
    if report.is_file():
        ctx.register(report, "metadata")

    result = {
        "engine": "reconstruction",
        "output_directory": "reconstruction",
        "model": "reconstruction/textured.obj",
        "assets": bridged,
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "resumed_from_checkpoint": summary.get("resumed_from_checkpoint"),
        "checkpoint": "reconstruction/project_state.json",
    }
    ctx.log(
        f"Reconstruction engine finished in "
        f"{summary.get('elapsed_seconds')}s; model ready at "
        f"reconstruction/textured.obj",
        stage="MESHROOM",
        operation="engine",
        status="completed",
    )
    return result


# --------------------------------------------------------------------------
def _reconstruct_external(ctx: PipelineContext) -> Dict[str, Any]:
    """Legacy path: run an external Meshroom/AliceVision executable."""
    # Meshroom distributions expose different command-line frontends.  The
    # configured executable is treated as a safe wrapper and must write a
    # newly changed model artifact under reconstruction/.
    ctx.check()
    frames = ctx.files(roots=["processed_frames"], kinds=["processed_frame"])
    if not frames:
        frames = ctx.files(roots=["frames"], kinds=["frame"])
    if not frames:
        raise StageError("MESHROOM FAILED: no extracted frames are available")
    executable = ctx.settings.tool("meshroom_path")
    if not executable:
        raise StageError(
            "MESHROOM FAILED: Meshroom/AliceVision is not installed or configured",
            exit_code=127,
            details={"install_hint": "Set MESHROOM_PATH or configure it in Settings"},
        )
    input_dir = ctx.project_dir / ("processed_frames" if ctx.project_dir.joinpath("processed_frames").exists() and frames and frames[0][1].parent.name == "processed_frames" else "frames")
    output_dir = ctx.project_dir / "reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)
    before = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    # Meshroom distributions expose different frontends.  Keep the command an
    # argument list and allow a site wrapper to provide the project-specific CLI.
    command = [executable, "--input", str(input_dir), "--output", str(output_dir)]
    try:
        run_command(
            command,
            cwd=ctx.project_dir,
            log_path=ctx.tool_log("meshroom"),
            control=ctx.control,
            timeout_seconds=ctx.config.command_timeout_seconds,
        )
    except CommandError as exc:
        raise StageError(
            "MESHROOM FAILED: reconstruction process returned an error",
            exit_code=exc.returncode,
            details={"command": exc.command, "output": exc.output[-6000:]},
        ) from exc
    model_suffixes = {".obj", ".fbx", ".glb", ".gltf", ".ply", ".stl"}
    produced_models = []
    for artifact in output_dir.rglob("*"):
        if not artifact.is_file():
            continue
        stat = artifact.stat()
        if artifact.suffix.lower() in model_suffixes and (
            artifact not in before
            or before[artifact] != (stat.st_mtime_ns, stat.st_size)
        ):
            produced_models.append(artifact)
    if not produced_models:
        raise StageError(
            "MESHROOM FAILED: the reconstruction command produced no new model artifact",
            exit_code=1,
        )
    for artifact in output_dir.rglob("*"):
        if not artifact.is_file():
            continue
        suffix = artifact.suffix.lower()
        if suffix in {".obj", ".fbx", ".glb", ".gltf", ".ply", ".stl"}:
            artifact_kind = "model"
        elif suffix in {".jpg", ".jpeg", ".png", ".webp", ".exr"}:
            artifact_kind = "texture"
        else:
            artifact_kind = "metadata"
        ctx.register(artifact, artifact_kind)
    result = {"input_directory": input_dir.name, "output_directory": "reconstruction"}
    ctx.log(
        "Meshroom reconstruction completed",
        stage="MESHROOM",
        operation="reconstruct",
        status="completed",
        external_command=" ".join(shlex.quote(part) for part in command),
        exit_code=0,
    )
    return result
