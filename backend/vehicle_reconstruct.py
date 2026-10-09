"""vehicle-reconstruct - command line front end for the CPU reconstruction engine.

    vehicle-reconstruct input.mp4 --cpu-only
    vehicle-reconstruct ./photos/ --threads 4 --frame-interval 0.25
    vehicle-reconstruct resume stinger_001
    vehicle-reconstruct status  stinger_001

This file only parses arguments and prints human-readable output; all work
lives in the ``reconstruction`` package so the web backend can call exactly
the same code path through the Pipeline Manager.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from reconstruction import (  # noqa: E402
    PRESETS,
    ReconConfig,
    ReconstructionError,
    run_pipeline,
    status as engine_status,
)

REPO_ROOT = BACKEND_DIR.parent
PROJECTS_DIR = REPO_ROOT / "projects"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vehicle-reconstruct",
        description="CPU-only 3D reconstruction of a vehicle from a video or "
                    "a directory of photographs (sparse point cloud today; "
                    "dense/mesh/texture/Blender/BUSSID stages follow).",
        epilog="CPU-only by design: no CUDA, no NVIDIA, no GPU fallback.",
    )
    parser.add_argument("target",
                        help="video file, directory of photographs, or a "
                             "projects/<id> directory (for resume/status)")
    parser.add_argument("--cpu-only", action="store_true", default=True,
                        help="required mode (default, and the only mode)")
    parser.add_argument("--no-cpu-only", action="store_false", dest="cpu_only",
                        help=argparse.SUPPRESS)  # refused loudly, never silent
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads (default 2, max 4 on this host)")
    parser.add_argument("--max-image-size", type=int, default=None,
                        help="longest image side fed to SIFT (default from "
                             "quality preset)")
    parser.add_argument("--features", type=int, default=None,
                        help="max SIFT features per image (default from preset)")
    parser.add_argument("--frame-interval", type=float, default=None,
                        help="seconds between extracted frames (default 0.3)")
    parser.add_argument("--quality", choices=sorted(PRESETS), default=None,
                        help="quality/memory preset (default medium, dense=max)")
    parser.add_argument("--target-triangles", type=int, default=None,
                        help="low-poly triangle budget (default from preset)")
    parser.add_argument("--front-azimuth", default=None,
                        help="degrees labeling the vehicle front for coverage "
                             "reporting, or 'unknown' to keep neutral labels "
                             "(default: first registered view)")
    parser.add_argument("--project", default=None,
                        help="project id to create/refresh (default: derived "
                             "from the input name)")
    parser.add_argument("--force", action="store_true",
                        help="re-run stages even if a checkpoint says "
                             "they are complete")
    parser.add_argument("--status", action="store_true",
                        help="print checkpoint status and exit (same as the "
                             "'status' command)")
    parser.add_argument("--json", action="store_true",
                        help="print machine-readable JSON instead of prose")
    return parser


def resolve_project(target: str) -> Path:
    """PROJECT_ID -> directory (absolute paths and bare ids both accepted)."""
    candidate = Path(target).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    for guess in (PROJECTS_DIR / target, PROJECTS_DIR / target.replace("-", "_")):
        if guess.is_dir():
            return guess.resolve()
    raise ReconstructionError(
        f"No project found at '{target}'.",
        suggestion=f"Known projects live in {PROJECTS_DIR}; run "
                   "`vehicle-reconstruct status <id>` after listing that "
                   "directory, or pass a new input to create one.",
    )


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "project"


def prepare_project(input_path: Path, requested: Optional[str]) -> Path:
    """Create projects/<id>/ and place the input inside input/ (section 28)."""
    if input_path.is_dir():
        default_name = input_path.resolve().name
        files = [p for p in sorted(input_path.iterdir())
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            raise ReconstructionError(
                f"{input_path} contains no supported images.",
                suggestion="Supported photograph formats: "
                           + ", ".join(sorted(IMAGE_EXTENSIONS)),
            )
    elif input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS:
        default_name = input_path.stem
        files = [input_path]
    else:
        raise ReconstructionError(
            f"'{input_path}' is neither a video nor a directory of "
            "photographs.",
            suggestion="Pass a video (%s) or a folder of photos (%s)."
                       % (", ".join(sorted(VIDEO_EXTENSIONS)),
                          ", ".join(sorted(IMAGE_EXTENSIONS))),
        )

    project_id = slugify(requested or default_name)
    project_dir = PROJECTS_DIR / project_id
    input_dir = project_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        destination = input_dir / source.name
        if destination.exists():
            continue
        try:
            os_link(source, destination)   # same filesystem -> free copy
        except OSError:
            shutil.copy2(source, destination)
    return project_dir


def os_link(source: Path, destination: Path) -> None:
    import os
    os.link(source, destination)


def make_config(args: argparse.Namespace, project_dir: Path) -> ReconConfig:
    if not args.cpu_only:
        raise ReconstructionError(
            "--no-cpu-only is refused: this host has no CUDA device.",
            suggestion="Run without that flag; CPU-only is the only supported "
                       "mode and every algorithm here is CPU-native.",
        )
    cfg = ReconConfig.for_project(
        project_dir,
        quality=args.quality,
        threads=args.threads,
        frame_interval=args.frame_interval,
        max_image_size=args.max_image_size,
        max_features=args.features,
        target_triangles=args.target_triangles,
        cpu_only=args.cpu_only,
    )
    if args.front_azimuth is not None:
        if args.front_azimuth.lower() == "unknown":
            cfg.front_azimuth = "unknown"
        else:
            try:
                cfg.front_azimuth = float(args.front_azimuth) % 360.0
            except ValueError:
                raise ReconstructionError(
                    f"--front-azimuth '{args.front_azimuth}' is not a number.",
                    suggestion="Pass degrees (0-360) or the word 'unknown'.",
                ) from None
    return cfg


def progress_printer(store: Optional[Dict[str, Any]] = None):
    def on_progress(event: Dict[str, Any]) -> None:
        stage = event.get("stage")
        state = event.get("status")
        if state == "running":
            print(f"  -> {stage}: running ...", flush=True)
        elif state == "skipped":
            print(f"  = {stage}: complete (checkpoint reused)", flush=True)
        elif state == "complete":
            print(f"  OK {stage}", flush=True)
        elif state == "failed":
            print(f"  !! {stage}: {event.get('error')}", file=sys.stderr,
                  flush=True)
    return on_progress


# ---------------------------------------------------------------------------
def cmd_status(cfg: ReconConfig, as_json: bool) -> int:
    snapshot = engine_status(cfg)
    if as_json:
        print(json.dumps(snapshot, indent=2))
        return 0
    print(f"Project : {snapshot['project']}")
    print(f"Path    : {snapshot['project_dir']}")
    print(f"CPU-only: {snapshot['cpu_only']}")
    print("Stages  :")
    for row in snapshot["stages"]:
        status = row["status"]
        marker = {"complete": "[done]", "running": "[run ]",
                  "failed": "[FAIL]", "skipped": "[skip]",
                  "pending": "[    ]"}.get(status, f"[{status}]")
        elapsed = (f"{row['elapsed_seconds']:.1f}s"
                   if row.get("elapsed_seconds") is not None else "")
        print(f"  {marker} {row['label']} {elapsed}")
        if row.get("failure_reason"):
            print(f"          reason: {row['failure_reason']}")
    if snapshot.get("summary"):
        print(f"Summary : {snapshot['summary']}")
    for message in snapshot.get("coverage_messages") or []:
        print(f"Coverage: {message}")
    if snapshot.get("exports"):
        print("Exports :")
        for export in snapshot["exports"]:
            print(f"  {snapshot['project_dir']}/{export}")
    return 0 if not any(r["status"] == "failed" for r in snapshot["stages"]) else 1


def cmd_run(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser()
    if not target.exists():
        project_dir = resolve_project(args.target)  # raises with guidance
        # A project directory without --force means "resume it".
        args.force = False
    else:
        project_dir = prepare_project(target, args.project)

    cfg = make_config(args, project_dir)
    if args.status:
        return cmd_status(cfg, args.json)

    started = time.time()
    if not args.json:
        print(f"CPU-ONLY MODE ENABLED  (project: {cfg.project_dir.name}, "
              f"threads: {cfg.threads}, quality: {cfg.quality})")
    try:
        summary = run_pipeline(cfg, force=args.force,
                               on_progress=None if args.json else progress_printer())
    except ReconstructionError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": exc.message,
                              "suggestion": exc.suggestion,
                              "details": exc.details}, indent=2, default=str))
        else:
            print(f"\nFAILED: {exc.message}", file=sys.stderr)
            if exc.suggestion:
                print(f"Suggestion: {exc.suggestion}", file=sys.stderr)
            print(f"Details: {cfg.state_path}", file=sys.stderr)
        return 1

    snapshot = engine_status(cfg)
    if args.json:
        print(json.dumps({"ok": True, "summary": summary,
                          "status": snapshot}, indent=2, default=str))
        return 0

    print(f"\nReconstruction finished in {time.time() - started:.1f}s "
          f"(CPU-only).")
    if snapshot.get("summary"):
        print(f"\n{snapshot['summary']}")
    for message in snapshot.get("coverage_messages") or []:
        print(f"  * {message}")
    print("\nInspectable outputs:")
    for export in snapshot.get("exports") or []:
        print(f"  {Path(snapshot['project_dir']) / export}")
    print(f"  checkpoint: {cfg.state_path}")
    print("\nNext: open sparse/sparse_point_cloud.ply in MeshLab/Blender to "
          "review the sparse reconstruction before the dense phase.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Sub-command forms from the spec: `resume ID` and `status ID`.
    mode = "run"
    if argv and argv[0] in {"resume", "status"}:
        mode = argv.pop(0)
    parser = build_parser()
    args = parser.parse_args(argv)

    if mode == "status":
        args.status = True
        args.force = False
    elif mode == "resume":
        args.force = False

    try:
        return cmd_run(args)
    except ReconstructionError as exc:
        print(f"FAILED: {exc.message}", file=sys.stderr)
        if exc.suggestion:
            print(f"Suggestion: {exc.suggestion}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted - checkpoints were written; run "
              "`vehicle-reconstruct resume <project>` to continue.",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
