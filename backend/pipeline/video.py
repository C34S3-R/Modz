"""Video inspection and FFmpeg helpers."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict

from .common import PipelineContext, StageError
from utils.subprocess import CommandError, run_capture_command


def _video(ctx: PipelineContext):
    found = ctx.input_file("video")
    if not found:
        raise StageError("VIDEO VALIDATION FAILED: no supported reference video was uploaded")
    return found[1]


def validate(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.check()
    video_entry = ctx.input_file("video")
    if not video_entry:
        images = ctx.input_files("image")
        if images:
            image_paths = [path for _row, path in images]
            source_report = _check_images(image_paths)
            metadata = {
                "mode": "images",
                "file": "multiple reference images",
                "image_count": len(images),
                "files": [row["path"] for row, _path in images],
                "source_check": source_report,
            }
            metadata_path = ctx.write_json("metadata", "inputs.json", metadata)
            ctx.register(metadata_path, "metadata")
            if not source_report["ok"]:
                if _force_allowed(ctx):
                    ctx.log(
                        "Source check FAILED but force_weak_source is on - "
                        "running anyway: " + "; ".join(source_report["failures"]),
                        stage="VIDEO",
                        operation="validate",
                        status="completed",
                        level="WARNING",
                    )
                else:
                    raise StageError(
                        "VIDEO VALIDATION FAILED: reference images do not meet "
                        "the quality bar - " + "; ".join(source_report["failures"])
                        + ". " + (source_report["guidance"] or ""),
                        details={"source_check": source_report},
                    )
            ctx.log(
                f"Validated {len(images)} reference images "
                f"(source check {source_report['score']})",
                stage="VIDEO",
                operation="validate",
                status="completed",
            )
            return metadata
    video = _video(ctx)
    if not video.is_file() or not video.stat().st_size:
        raise StageError("VIDEO VALIDATION FAILED: the reference video is missing or empty")
    ffprobe = ctx.settings.tool("ffprobe_path")
    if not ffprobe:
        raise StageError(
            "VIDEO VALIDATION FAILED: ffprobe is not installed or configured",
            exit_code=127,
        )
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(video),
    ]
    rendered = " ".join(shlex.quote(part) for part in command)
    try:
        result = run_capture_command(
            command,
            cwd=ctx.project_dir,
            log_path=ctx.tool_log("video"),
            control=ctx.control,
            timeout_seconds=ctx.config.command_timeout_seconds,
        )
    except CommandError as exc:
        raise StageError(
            "VIDEO VALIDATION FAILED: ffprobe rejected the video",
            exit_code=exc.returncode,
            details={"output": exc.output[-4000:]},
        ) from exc
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise StageError("VIDEO VALIDATION FAILED: ffprobe returned invalid metadata") from exc
    streams = payload.get("streams") or []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video_stream:
        raise StageError("VIDEO VALIDATION FAILED: no video stream was found")
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise StageError("VIDEO VALIDATION FAILED: video resolution is invalid")
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1")
    duration = float(payload.get("format", {}).get("duration") or 0)
    frame_count = int(video_stream.get("nb_frames") or 0)
    if frame_count <= 0 and fps > 0 and duration > 0:
        frame_count = int(fps * duration)
    metadata = {
        "file": video.name,
        "codec": video_stream.get("codec_name"),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "duration_seconds": round(duration, 3),
        "frame_count": frame_count,
    }
    source_report = _check_video(video, ctx)
    metadata["source_check"] = source_report
    metadata_path = ctx.write_json("metadata", "video.json", metadata)
    ctx.register(metadata_path, "metadata")
    if not source_report["ok"]:
        if _force_allowed(ctx):
            ctx.log(
                "Source check FAILED but force_weak_source is on - "
                "running anyway: " + "; ".join(source_report["failures"]),
                stage="VIDEO",
                operation="validate",
                status="completed",
                level="WARNING",
            )
        else:
            raise StageError(
                "VIDEO VALIDATION FAILED: the reference video does not meet the "
                "quality bar - " + "; ".join(source_report["failures"])
                + ". " + (source_report["guidance"] or ""),
                details={"source_check": source_report},
            )
    ctx.log(
        f"Validated {video.name}: {width}x{height}, {metadata['fps']} fps, "
        f"{metadata['duration_seconds']}s "
        f"(source check {source_report['score']})",
        stage="VIDEO",
        operation="validate",
        status="completed",
        external_command=rendered,
        exit_code=0,
    )
    return metadata


def _force_allowed(ctx: PipelineContext) -> bool:
    """Explicit per-project override: run even when the source check fails."""
    return bool((ctx.project.get("options") or {}).get("force_weak_source", False))


def _parse_fps(value: str) -> float:
    try:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _check_video(video: Path, ctx: PipelineContext) -> Dict[str, Any]:
    from utils.source_check import check_video_source

    try:
        return check_video_source(
            video, ffprobe=ctx.settings.tool("ffprobe_path")
        )
    except Exception as exc:
        return {
            "ok": True,
            "score": "n/a",
            "checks": [],
            "failures": [],
            "warnings": [f"source sampler skipped: {exc}"],
            "guidance": None,
            "meta": {},
        }


def _check_images(paths: list) -> Dict[str, Any]:
    from utils.source_check import check_image_set

    try:
        return check_image_set([Path(p) for p in paths])
    except Exception as exc:
        return {
            "ok": True,
            "score": "n/a",
            "checks": [],
            "failures": [],
            "warnings": [f"source sampler skipped: {exc}"],
            "guidance": None,
            "meta": {},
        }
