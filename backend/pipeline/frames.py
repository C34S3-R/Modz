"""Frame extraction and conservative frame filtering."""

from __future__ import annotations

import hashlib
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict

from .common import PipelineContext, StageError
from utils.subprocess import CommandError, run_command


def extract(ctx: PipelineContext) -> Dict[str, Any]:
    # Extraction never edits the uploaded video/images.  It creates new frame
    # files that later stages can safely filter or preprocess.
    ctx.check()
    video_entry = ctx.input_file("video")
    if not video_entry:
        images = ctx.input_files("image")
        if not images:
            raise StageError("FRAME EXTRACTION FAILED: no reference video or images are available")
        output = ctx.project_dir / "frames"
        output.mkdir(parents=True, exist_ok=True)
        for old in output.glob("frame_*.*"):
            try:
                old.unlink()
            except OSError:
                pass
        copied = []
        for index, (_row, source) in enumerate(images, start=1):
            ctx.check()
            destination = output / f"frame_{index:06d}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            ctx.register(destination, "frame")
            copied.append(destination)
        if not copied:
            raise StageError("FRAME EXTRACTION FAILED: no reference images were available", exit_code=1)
        ctx.log(
            f"Prepared {len(copied)} reference images as frame inputs",
            stage="FRAMES",
            operation="extract",
            status="completed",
        )
        return {"directory": "frames", "count": len(copied), "source": "images"}
    video = video_entry[1]
    ffmpeg = ctx.settings.tool("ffmpeg_path")
    if not ffmpeg:
        raise StageError("FRAME EXTRACTION FAILED: ffmpeg is not installed or configured", exit_code=127)
    output = ctx.project_dir / "frames"
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("frame_*.*"):
        try:
            old.unlink()
        except OSError:
            pass
    fps = ctx.project.get("options", {}).get("fps", 4)
    maximum = ctx.project.get("options", {}).get("max_frames", 300)
    quality = ctx.project.get("options", {}).get("quality", 2)
    image_format = str(ctx.project.get("options", {}).get("image_format", "jpg")).lower().lstrip(".")
    if image_format not in {"jpg", "jpeg", "png"}:
        image_format = "jpg"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(maximum),
        "-q:v",
        str(quality),
        str(output / f"frame_%06d.{image_format}"),
    ]
    try:
        run_command(
            command,
            cwd=ctx.project_dir,
            log_path=ctx.tool_log("frames"),
            control=ctx.control,
            timeout_seconds=ctx.config.command_timeout_seconds,
        )
    except CommandError as exc:
        raise StageError(
            "FRAME EXTRACTION FAILED: ffmpeg returned an error",
            exit_code=exc.returncode,
            details={"command": exc.command, "output": exc.output},
        ) from exc
    files = sorted(output.glob(f"frame_*.{image_format}"))
    if not files:
        raise StageError("FRAME EXTRACTION FAILED: ffmpeg produced no frames", exit_code=1)
    for path in files:
        ctx.register(path, "frame")
    ctx.log(
        f"Extracted {len(files)} frames at {fps} fps (maximum {maximum})",
        stage="FRAMES",
        operation="extract",
        status="completed",
        external_command=" ".join(shlex.quote(part) for part in command),
        exit_code=0,
    )
    return {"directory": "frames", "count": len(files), "fps": fps, "format": image_format}


def filter_frames(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.check()
    frames_dir = ctx.project_dir / "frames"
    rejected_dir = ctx.project_dir / "rejected_frames"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[Path] = []
    rejected: list[Path] = []
    seen_hashes: set[str] = set()
    for path in sorted(frames_dir.glob("frame_*")):
        ctx.check()
        if not path.is_file() or path.stat().st_size == 0:
            rejected.append(path)
            continue
        try:
            digest_builder = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest_builder.update(chunk)
            digest = digest_builder.hexdigest()
        except OSError:
            rejected.append(path)
            continue
        if digest in seen_hashes:
            rejected.append(path)
            continue
        seen_hashes.add(digest)
        accepted.append(path)
    for path in rejected:
        if path.exists():
            shutil.move(str(path), str(rejected_dir / path.name))
            ctx.register(rejected_dir / path.name, "rejected_frame")
    for path in accepted:
        ctx.register(path, "frame")
    if not accepted:
        raise StageError("FRAME FILTERING FAILED: no usable frames remained", exit_code=1)
    result = {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_directory": "frames",
        "rejected_directory": "rejected_frames",
    }
    ctx.write_json("frames", "filtering.json", result)
    ctx.register(ctx.project_dir / "frames" / "filtering.json", "metadata")
    ctx.log(
        f"Kept {len(accepted)} frames; moved {len(rejected)} duplicate/empty frames to rejected_frames",
        stage="FILTER",
        operation="filter",
        status="completed",
    )
    return result


def preprocess(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.check()
    source_dir = ctx.project_dir / "frames"
    output_dir = ctx.project_dir / "processed_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(source_dir.glob("frame_*"))
    if not files:
        raise StageError("PREPROCESSING FAILED: no accepted frames are available")
    for old in output_dir.glob("frame_*"):
        try:
            old.unlink()
        except OSError:
            pass
    for source in files:
        ctx.check()
        shutil.copy2(source, output_dir / source.name)
    for path in output_dir.glob("frame_*"):
        ctx.register(path, "processed_frame")
    result = {"count": len(files), "directory": "processed_frames"}
    ctx.log(
        f"Prepared {len(files)} frames without modifying the extracted originals",
        stage="PREPROCESS",
        operation="copy",
        status="completed",
    )
    return result
