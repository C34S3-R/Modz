"""PHASE 1 - input validation, frame extraction, frame quality analysis.

Flow (spec sections 4 and 5):

    input/*.mp4  --ffmpeg-->  frames/original/frame_XXXXXX.jpg
                             (never modified, never deleted)
    frames/original/*       --quality metrics-->
                             frames/selected/*   (accepted copies)
                             frames/rejected/*   (copies + reasons)

Design notes for this machine:
  * FFmpeg streams the video; the file is never loaded into RAM.
  * Only the frames/original directory is read one file at a time.
  * Rejection is deliberately lenient - a slightly imperfect frame may hold
    the only viewpoint of the matatu's rear, so thresholds reject only the
    extreme (near-black, smeared, byte-identical) cases and every rejection
    records *why*.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ReconConfig, ReconstructionError
from .state import StageContext, StateStore, memory_pressure
from utils.subprocess import CommandError, run_capture_command, run_command

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _cv2():
    import cv2
    return cv2


def find_input(cfg: ReconConfig) -> Tuple[str, List[Path]]:
    """Locate the project input: one video, or a directory of photographs."""
    input_dir = cfg.input_dir
    if not input_dir.is_dir():
        raise ReconstructionError(
            f"No input directory at {input_dir}",
            suggestion="Create the project with an input/ folder containing "
                       "reference video or photographs.",
        )
    videos = sorted(p for p in input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    images = sorted(p for p in input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if videos:
        return "video", videos
    if images:
        return "image", images
    raise ReconstructionError(
        f"No usable input found in {input_dir}",
        suggestion="Put a video (%s) or photographs (%s) into the input "
                   "directory."
                   % (", ".join(sorted(VIDEO_EXTENSIONS)),
                      ", ".join(sorted(IMAGE_EXTENSIONS))),
        details={"directory": str(input_dir)},
    )


# --------------------------------------------------------------------------
def validate_input(cfg: ReconConfig, store: StateStore, ctx: StageContext) -> Dict[str, Any]:
    """ffprobe the video (or inspect the photos) before any heavy work."""
    kind, paths = find_input(cfg)
    meta: Dict[str, Any] = {"input_type": kind, "files": [p.name for p in paths]}

    if kind == "video":
        video = paths[0]
        ffprobe = shutil.which("ffprobe") or "ffprobe"
        command = [
            ffprobe, "-v", "error",
            "-show_entries",
            "format=duration,size,format_name:"
            "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames",
            "-of", "json", str(video),
        ]
        log_path = cfg.d("logs", "ffprobe.log")
        try:
            # Capturing variant: run_command streams to the log but returns an
            # empty stdout, which would silently read as "no video stream".
            proc = run_capture_command(
                command, cwd=cfg.project_dir, log_path=log_path,
                timeout_seconds=300,
            )
        except CommandError as exc:
            raise ReconstructionError(
                "Input validation failed: ffprobe could not read the video "
                f"({video.name}).",
                suggestion="Verify the file is a valid video; re-export it "
                           "with FFmpeg if the container is corrupt.",
                details={"exit_code": exc.returncode, "output": exc.output[-800:]},
                exit_code=exc.returncode,
            ) from exc
        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            payload = {}
        fmt = payload.get("format") or {}
        streams = payload.get("streams") or []
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if not video_stream:
            raise ReconstructionError(
                f"{video.name} contains no video stream.",
                suggestion="Provide a video with a picture track, or a "
                           "directory of photographs instead.",
                details={"streams": [s.get("codec_type") for s in streams]},
            )
        try:
            duration = float(fmt.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        fps = _parse_rate(video_stream.get("r_frame_rate"))
        if duration <= 0 and fps > 0 and video_stream.get("nb_frames"):
            try:
                duration = int(video_stream["nb_frames"]) / fps
            except (TypeError, ValueError, ZeroDivisionError):
                duration = 0.0
        if duration <= 0:
            raise ReconstructionError(
                f"{video.name} reports no duration.",
                suggestion="Re-mux with `ffmpeg -i in.mp4 -c copy out.mp4` and "
                           "upload the repaired file.",
            )
        meta.update({
            "file": video.name,
            "size_bytes": int(fmt.get("size") or video.stat().st_size),
            "duration_seconds": round(duration, 3),
            "codec": video_stream.get("codec_name"),
            "width": int(video_stream.get("width") or 0),
            "height": int(video_stream.get("height") or 0),
            "fps": round(fps, 3),
        })
        # Work out an extraction plan that respects max_frames: if the video
        # is longer than max_frames * interval, widen the interval instead of
        # silently dropping the tail of the vehicle orbit.
        planned = int(math.floor(duration / cfg.frame_interval)) + 1
        effective_interval = cfg.frame_interval
        if planned > cfg.max_frames:
            effective_interval = round(duration / cfg.max_frames, 3)
            effective_interval = max(0.05, effective_interval)
        meta["planned_frames"] = min(planned, cfg.max_frames)
        meta["effective_interval_seconds"] = effective_interval
        ctx.record["result"] = meta
        (cfg.d("metadata")).mkdir(parents=True, exist_ok=True)
        (cfg.d("metadata") / "input.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    # Photograph set: check each file opens before we commit to the pipeline.
    cv2 = _cv2()
    readable = 0
    unreadable: List[str] = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:
            unreadable.append(path.name)
        else:
            readable += 1
    if readable < 3:
        raise ReconstructionError(
            f"Only {readable} of {len(paths)} photographs could be decoded.",
            suggestion="Structure-from-Motion needs at least 3 overlapping "
                       "views of the vehicle; re-export the images as JPEG or PNG.",
            details={"unreadable": unreadable[:20]},
        )
    meta.update({"count": len(paths), "readable": readable,
                 "unreadable": unreadable, "effective_interval_seconds": 0})
    ctx.record["result"] = meta
    return meta


def _parse_rate(value: Optional[str]) -> float:
    try:
        if not value:
            return 0.0
        if "/" in value:
            num, _, den = value.partition("/")
            num_f, den_f = float(num), float(den)
            return num_f / den_f if den_f else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------
def extract_frames(cfg: ReconConfig, store: StateStore, ctx: StageContext,
                   meta: Dict[str, Any]) -> Dict[str, Any]:
    """Stream the video through FFmpeg into frames/original/ (section 4)."""
    original = cfg.d("frames", "original")
    original.mkdir(parents=True, exist_ok=True)
    # Fresh run: remove only frames we generated, never anything in input/.
    for stale in original.glob("frame_*.jpg"):
        try:
            stale.unlink()
        except OSError:
            pass

    if meta.get("input_type") == "image":
        # Count only images: non-image files (a gt.json, a README) must not
        # consume a frame number or the frame/source mapping shifts.
        copied = 0
        manifest = []
        for path in sorted(cfg.input_dir.glob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            destination = original / f"frame_{copied + 1:06d}{path.suffix.lower()}"
            shutil.copy2(path, destination)
            manifest.append({"file": destination.name, "source": path.name})
            copied += 1
        result = {"directory": "frames/original", "count": copied, "source": "images"}
        (cfg.d("frames") / "extraction_manifest.json").write_text(
            json.dumps({"interval_seconds": None, "frames": manifest},
                       indent=2),
            encoding="utf-8",
        )
        ctx.result(result)
        return result

    kind, paths = find_input(cfg)
    video = paths[0]
    interval = float(meta.get("effective_interval_seconds") or cfg.frame_interval)
    quality = max(2, min(31, int(cfg.jpeg_quality)))
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vf", f"fps=1/{interval:.4f}",
        "-frames:v", str(cfg.max_frames),
        "-q:v", str(quality),
        str(original / "frame_%06d.jpg"),
    ]
    ctx.set_command(command)
    log_path = cfg.d("logs", "frame_extraction.log")
    try:
        run_command(command, cwd=cfg.project_dir, log_path=log_path,
                    timeout_seconds=3600)
    except CommandError as exc:
        raise ReconstructionError(
            "Frame extraction failed: ffmpeg returned an error while reading "
            f"{video.name}.",
            suggestion="Check logs/frame_extraction.log; if the video plays in "
                       "a player, re-encode it (`ffmpeg -i in.mp4 -c:v libx264 out.mp4`).",
            details={"exit_code": exc.returncode, "command": command,
                     "output": (exc.output or "")[-800:]},
            exit_code=exc.returncode,
        ) from exc

    files = sorted(original.glob("frame_*.jpg"))
    if not files:
        raise ReconstructionError(
            "Frame extraction produced no frames.",
            suggestion="The video may be zero-length or all-black; inspect it "
                       "with ffprobe and re-upload.",
            details={"command": command},
            exit_code=1,
        )
    ctx.set_output(exit_code=0)
    # Timestamps are approximate but stable: frame N sits at (N-1) * interval.
    manifest = [
        {"file": f.name, "time_seconds": round(i * interval, 3)}
        for i, f in enumerate(files)
    ]
    cfg.d("frames").mkdir(parents=True, exist_ok=True)
    (cfg.d("frames") / "extraction_manifest.json").write_text(
        json.dumps({"interval_seconds": interval, "frames": manifest}, indent=2),
        encoding="utf-8",
    )
    result = {"directory": "frames/original", "count": len(files),
              "interval_seconds": interval, "source": "video"}
    ctx.result(result)
    ctx.add_output_files(files[:5])
    return result


# --------------------------------------------------------------------------
def _metrics(path: Path) -> Dict[str, Any]:
    """Cheap CPU metrics for one frame; nothing is held after the call."""
    cv2 = _cv2()
    # Reduce very large frames before analysis - metrics do not need 4K.
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return {"readable": False}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    thumb = small.astype("float32").ravel()
    thumb = (thumb - thumb.mean()) / (float(thumb.std()) + 1e-6)
    bits = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).ravel()
    ahash = bits > bits.mean()
    return {
        "readable": True,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "blur_score": round(blur, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "thumbnail": thumb,
        "ahash": ahash,
    }


def _hamming(a, b) -> int:
    return int((a != b).sum())


def select_frames(cfg: ReconConfig, store: StateStore, ctx: StageContext) -> Dict[str, Any]:
    """Quality-analyse every original frame and split selected/rejected.

    Writes frames/selected/, frames/rejected/ and frames/rejected_manifest.json
    with one entry per rejection and its reasons (section 5).
    """
    original = cfg.d("frames", "original")
    selected_dir = cfg.d("frames", "selected")
    rejected_dir = cfg.d("frames", "rejected")
    selected_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(original.glob("frame_*.jpg"))
    if not frames:
        raise ReconstructionError(
            "Frame selection found no extracted frames.",
            suggestion="Run frame extraction first (stage order handles this); "
                       "if it ran, check that ffmpeg wrote to frames/original.",
        )

    accepted_names: List[str] = []
    rejected: List[Dict[str, Any]] = []
    accepted_thumb: Dict[str, Any] = {}
    accepted_hash: Dict[str, Any] = {}
    pressure_samples: List[Dict[str, Any]] = []

    for index, path in enumerate(frames, start=1):
        ctx.check()
        m = _metrics(path)
        if not m.get("readable"):
            rejected.append({"file": path.name, "reasons": ["unreadable"],
                             "metrics": {}})
            _copy_into(path, rejected_dir / path.name)
            continue
        reasons: List[str] = []
        if m["blur_score"] < cfg.min_blur_score:
            reasons.append(
                f"blurry (laplacian variance {m['blur_score']:.1f} < "
                f"{cfg.min_blur_score})")
        if m["brightness"] < cfg.min_brightness:
            reasons.append(
                f"too dark (mean {m['brightness']:.0f} < {cfg.min_brightness})")
        elif m["brightness"] > cfg.max_brightness:
            reasons.append(
                f"overexposed (mean {m['brightness']:.0f} > {cfg.max_brightness})")

        # Near-duplicate check only against frames already accepted, and only
        # when *both* independent signals agree (identical 64-bit aHash AND
        # near-perfect thumbnail correlation).  A single weak signal would
        # throw away distinct viewpoints of the same vehicle - frames whose
        # background dominates the hash can look alike while showing
        # completely different geometry.  Section 5: do not reject
        # aggressively; a slightly imperfect frame may be the only view of
        # the rear of the matatu.
        if not reasons and accepted_thumb:
            best_name, best_corr = None, -1.0
            for name, thumb in accepted_thumb.items():
                corr = float((m["thumbnail"] * thumb).mean())
                if corr > best_corr:
                    best_corr, best_name = corr, name
            dist = _hamming(m["ahash"], accepted_hash[best_name]) \
                if best_name in accepted_hash else 99
            if (best_corr >= cfg.duplicate_corr_max
                    and dist <= cfg.duplicate_hamming_max):
                reasons.append(
                    f"near-duplicate of {best_name} "
                    f"(correlation {best_corr:.4f} >= "
                    f"{cfg.duplicate_corr_max}, aHash distance {dist} <= "
                    f"{cfg.duplicate_hamming_max})")

        if reasons:
            rejected.append({
                "file": path.name,
                "reasons": reasons,
                "metrics": {k: m[k] for k in
                            ("blur_score", "brightness", "contrast", "width", "height")},
            })
            _copy_into(path, rejected_dir / path.name)
        else:
            _copy_into(path, selected_dir / path.name)
            accepted_names.append(path.name)
            accepted_hash[path.name] = m["ahash"]
            accepted_thumb[path.name] = m["thumbnail"]

        if index % 10 == 0:
            pressure = memory_pressure(cfg.ram_limit_gb)
            if pressure["over_limit"] or pressure["critical"]:
                pressure_samples.append({"at_frame": index, **pressure})

    if len(accepted_names) < 3:
        raise ReconstructionError(
            f"Only {len(accepted_names)} of {len(frames)} frames passed quality "
            "checks - not enough for 3D reconstruction.",
            suggestion="Relax the thresholds (min_blur_score, min_brightness) or "
                       "supply a sharper/longer reference video.",
            details={"accepted": len(accepted_names), "total": len(frames),
                     "sample_rejections": rejected[:10]},
        )

    reason_counts: Dict[str, int] = {}
    for item in rejected:
        for reason in item["reasons"]:
            key = reason.split(" (")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    manifest = {
        "total_original": len(frames),
        "selected": len(accepted_names),
        "rejected": len(rejected),
        "rejection_reasons": reason_counts,
        "thresholds": {
            "min_blur_score": cfg.min_blur_score,
            "min_brightness": cfg.min_brightness,
            "max_brightness": cfg.max_brightness,
            "duplicate_hamming_max": cfg.duplicate_hamming_max,
            "duplicate_corr_max": cfg.duplicate_corr_max,
        },
        "rejected_frames": rejected,
        "memory_pressure_samples": pressure_samples,
    }
    (cfg.d("frames") / "rejected_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    # Keep the working set small enough for 6 GB: never feed more than
    # max_frames images to feature extraction even if extraction was generous.
    used = accepted_names
    dropped: List[str] = []
    if len(used) > cfg.max_frames:
        step = len(used) / cfg.max_frames
        keep = {used[int(i * step)] for i in range(cfg.max_frames)}
        dropped = [n for n in used if n not in keep]
        for name in dropped:
            target = selected_dir / name
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass
        used = sorted(keep)
    result = {
        "directory": "frames/selected",
        "selected": len(used),
        "rejected": len(rejected),
        "capped_by_max_frames": len(dropped),
        "rejection_reasons": reason_counts,
        "manifest": "frames/rejected_manifest.json",
    }
    ctx.result(result)
    return result


def _copy_into(source: Path, destination: Path) -> None:
    """Hard-link when possible (same filesystem) else copy. Originals stay."""
    if destination.exists():
        return
    try:
        import os
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
