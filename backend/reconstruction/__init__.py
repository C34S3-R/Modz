"""CPU-only vehicle reconstruction engine (Phases 1-3 delivered here).

This package is the "Reconstruction Engine" underneath the existing Pipeline
Manager:

    Browser -> FastAPI -> Job Manager -> Pipeline Manager -> Reconstruction
    Engine -> (later) Blender -> BUSSID Export

It is deliberately importable without FastAPI so the CLI
(``vehicle-reconstruct``) can drive it on its own.  The web backend calls the
same three entry points:

    run_pipeline(cfg, ...)   run/resume every checkpointed stage
    resume(...)              alias of run_pipeline (checkpoints decide)
    status(cfg)              read-only snapshot for the UI

Stages implemented so far, per the phasing plan:

    Phase 1  input_validation, frame_extraction, frame_selection
    Phase 2  feature_extraction (SIFT), feature_matching (sequential + RANSAC)
    Phase 3  sfm (poses, bundle adjustment, sparse cloud, reports)
    Phase 4  dense (plane-sweep depth maps + fused point cloud)
    Phase 5  mesh (TSDF fusion + marching cubes + vertex colours)
    Phase 6  lowpoly (QEM decimation to the preset's triangle budget)
    Phase 7  texture (hand-rolled UV atlas + frame bake)
    Phase 8  blender (optional headless cleanup, logged model-copy fallback)
    Phase 9  bussid (bussid/ prep + manifest + atomic .bussidmod package)

Phase 10 landed: the web pipeline's ``meshroom`` stage drives this same
engine (``run_pipeline(..., through="texture")``, see
``backend/pipeline/meshroom.py``); every stage key exists in
``state.STAGE_ORDER`` so the checkpoint file always shows the whole plan
with honest statuses.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import (  # noqa: F401  (re-exported API)
    CPU_ALGORITHMS,
    PRESETS,
    ReconConfig,
    ReconstructionError,
    enforce_cpu_only,
)

STAGE_PIPELINE: tuple[tuple[str, Callable], ...] = (
    ("input_validation", None),      # resolved lazily to avoid import cycles
    ("frame_extraction", None),
    ("frame_selection", None),
    ("feature_extraction", None),
    ("feature_matching", None),
    ("sfm", None),
    ("masking", None),
    ("dense", None),
    ("feature_matching", None),
    ("sfm", None),
    ("dense", None),
    ("mesh", None),
    ("lowpoly", None),
    ("texture", None),
    ("blender", None),
    ("bussid", None),
)


def _handlers() -> Dict[str, Callable]:
    from . import (bussid, blender, dense, features, frames, lowpoly,
                   masking, matching, mesh, sfm, texture)

    return {
        "input_validation": _stage_input_validation,
        "frame_extraction": _stage_frame_extraction,
        "frame_selection": frames.select_frames,
        "masking": masking.run_masking,
        "feature_extraction": features.extract_features,
        "feature_matching": matching.extract_matches,
        "sfm": sfm.run_sparse_multistart,
        "dense": dense.run_dense,
        "mesh": mesh.run_mesh,
        "lowpoly": lowpoly.run_lowpoly,
        "texture": texture.run_texture,
        "blender": blender.run_blender,
        "bussid": bussid.run_bussid,
    }


def _stage_input_validation(cfg, store, ctx):
    return frames.validate_input(cfg, store, ctx)


def _stage_frame_extraction(cfg, store, ctx):
    meta = store.stage_record("input_validation").get("result") or {}
    if not meta:
        # Validate on demand: the caller may have skipped straight to resume.
        meta = frames.validate_input(cfg, store, ctx) or {}
    return frames.extract_frames(cfg, store, ctx, meta)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def _frames_digest(cfg: "ReconConfig") -> str:
    """Digest of the selected frame set (names + sizes).

    Feature and match stages are per frame/pair, so they must be redone when
    frame selection changes - otherwise a stage would be skipped with a
    checkpoint while its inputs have moved on underneath it.
    """
    import hashlib
    selected = cfg.d("frames", "selected")
    if not selected.is_dir():
        return ""
    digest = hashlib.sha1()
    seen = 0
    try:
        for path in sorted(selected.glob("*.jpg")):
            info = path.stat()
            digest.update(f"{path.name}:{info.st_size};".encode("utf-8"))
            seen += 1
    except OSError:
        return ""
    # An empty (freshly created) directory means "no frames yet", which must
    # not be mistaken for a stable digest of nothing.
    return digest.hexdigest()[:16] if seen else ""


def _stage_fingerprint(cfg: "ReconConfig", key: str,
                       frames_digest: str) -> Optional[str]:
    """Fingerprint for one stage, qualified by the frames it reads.

    Both the comparison *and* the record must go through this single rule:
    when they disagree the stage is re-run on every resume.  That already cost
    one wasted matching pass, and an asymmetric exclusion would have turned it
    into a permanent one.

    ``frame_extraction`` is deliberately excluded - the digest covers
    ``frames/selected``, which is its downstream output and does not exist
    while it runs.
    """
    base = cfg.stage_fingerprint(key)
    if base is None or not frames_digest or key == "frame_extraction":
        return base
    return f"{base}|frames={frames_digest}"


def run_pipeline(
    cfg: ReconConfig,
    *,
    force: bool = False,
    control: Any = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    through: Optional[str] = None,
) -> Dict[str, Any]:
    """Run (or resume) every stage, honouring checkpoints.

    ``force=True`` re-runs stages that are already complete; by default a
    complete stage is never repeated, which is what makes a shutdown during
    dense reconstruction cheap to recover from (section 24).

    ``through`` stops after that stage key.  The web pipeline's
    ``meshroom`` stage runs the engine through ``texture`` and then keeps
    owning blender/lights/bussid/export with its own stages; the standalone
    CLI leaves it ``None`` and runs the full plan.
    """
    from .state import StateStore

    cfg.ensure_layout()
    algorithms = enforce_cpu_only(cfg.cpu_only)

    pipeline = list(STAGE_PIPELINE)
    if through is not None:
        keys = [key for key, _ in pipeline]
        if through not in keys:
            raise ReconstructionError(
                f"Unknown stage key for 'through': {through!r}",
                suggestion=f"Valid keys: {', '.join(keys)}",
            )
        pipeline = pipeline[:keys.index(through) + 1]

    store = StateStore(cfg.state_path)
    store.state["config"] = cfg.to_dict()
    store.state["cpu_only"] = True
    store.state["cpu_algorithms"] = algorithms
    store.save()

    log_path = cfg.d("logs", "pipeline.log")
    _log(log_path, "CPU-ONLY MODE ENABLED")
    for line in algorithms:
        _log(log_path, f"  algorithm: {line}")
    _log(log_path, f"project={cfg.project_dir} quality={cfg.quality} "
                   f"threads={cfg.threads} interval={cfg.frame_interval}s")

    handlers = _handlers()
    started = time.time()
    results: Dict[str, Any] = {}
    resumed: List[str] = []
    frames_digest = _frames_digest(cfg)

    for key, _ in pipeline:
        if key not in handlers:
            continue
        fingerprint = _stage_fingerprint(cfg, key, frames_digest)
        if not force and store.is_complete(key):
            stored = store.stage_record(key).get("fingerprint")
            if fingerprint is None or stored == fingerprint:
                resumed.append(key)
                results[key] = store.stage_record(key).get("result") or {}
                _log(log_path, f"stage {key}: complete (checkpoint reused)")
                if on_progress:
                    on_progress({"stage": key, "status": "skipped",
                                 "reason": "checkpoint"})
                continue
            _log(log_path,
                 f"stage {key}: settings or inputs changed, re-running")
        if on_progress:
            on_progress({"stage": key, "status": "running"})
        _log(log_path, f"stage {key}: starting")
        try:
            with store.stage(key, control=control) as ctx:
                result = handlers[key](cfg, store, ctx)
                results[key] = result or ctx.record.get("result") or {}
                if fingerprint is not None:
                    # Remembered so the next run can tell stale work apart
                    # from work that is still valid (section 24).  The frames
                    # digest is re-read here instead of reusing the one taken
                    # before the pipeline started: on a first run that value is
                    # empty (no frames exist yet), so every stage recorded a
                    # fingerprint without the suffix and was re-run - with
                    # everything downstream - on the very next resume.  On a
                    # fresh project that cost a full extra matching pass
                    # (~10 min) for work that had not changed.  Going through
                    # _stage_fingerprint keeps the stored value byte-identical
                    # to what the next run will compare against.
                    ctx.record["fingerprint"] = _stage_fingerprint(
                        cfg, key, _frames_digest(cfg))
        except ReconstructionError:
            _log(log_path, f"stage {key}: FAILED - "
                           f"{store.stage_record(key).get('failure_reason')}",
                 status="error")
            if on_progress:
                on_progress({"stage": key, "status": "failed",
                             "error": store.stage_record(key).get("failure_reason")})
            raise
        else:
            _log(log_path, f"stage {key}: complete "
                           f"({ctx.record.get('elapsed_seconds')}s)")
            if on_progress:
                on_progress({"stage": key, "status": "complete",
                             "result": results[key]})

    summary = {
        "project": cfg.project_dir.name,
        "project_dir": str(cfg.project_dir),
        "cpu_only": True,
        "resumed_from_checkpoint": resumed,
        "stages": store.summary(),
        "report": str(cfg.report_path) if cfg.report_path.is_file() else None,
        "state": str(cfg.state_path),
        "elapsed_seconds": round(time.time() - started, 1),
        "results": {k: v for k, v in results.items() if k == "sfm"},
    }
    store.state["last_run"] = {k: summary[k] for k in
                               ("project", "cpu_only", "elapsed_seconds")}
    store.save()
    _log(log_path, f"pipeline complete in {summary['elapsed_seconds']}s")
    return summary


def resume(cfg: ReconConfig, **kwargs: Any) -> Dict[str, Any]:
    """Resume a project from its checkpoint file (section 24)."""
    return run_pipeline(cfg, **kwargs)


def status(cfg: ReconConfig) -> Dict[str, Any]:
    """Read-only status snapshot: stages, report, exports (section 27)."""
    from .state import StateStore

    store = StateStore(cfg.state_path)
    report = None
    if cfg.report_path.is_file():
        try:
            import json
            report = json.loads(cfg.report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = None
    return {
        "project": cfg.project_dir.name,
        "project_dir": str(cfg.project_dir),
        "cpu_only": True,
        "stages": store.summary(),
        "summary": (report or {}).get("summary"),
        "exports": (report or {}).get("exports"),
        "coverage_messages": ((report or {}).get("coverage") or {}).get("messages"),
        "report": str(cfg.report_path) if cfg.report_path.is_file() else None,
        "state_file": str(cfg.state_path),
    }


# ---------------------------------------------------------------------------
def _log(path: Path, message: str, status: str = "info") -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} [{status.upper()}] {message}\n")
    except OSError:
        pass
