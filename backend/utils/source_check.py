"""Good-source verification for reference videos and image sets.

A reconstruction can only rebuild what the input actually observes, so a
short, tiny, or smeared clip produces a partial model (e.g. only the back
of the vehicle) after minutes of compute.  This gate checks the input
*before* the heavy stages run:

* video: duration, resolution, frame rate, file size, plus a sampled
  sharpness/brightness pass over evenly spaced frames;
* image sets: frame count, readability, and resolution spread.

Thresholds are env-overridable (see .env.example).  Failures block the
pipeline with guidance to supply more/better videos or images; warnings
are reported but do not block.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


MIN_DURATION_S = _f("SOURCE_MIN_DURATION_S", 10.0)
MIN_WIDTH = _i("SOURCE_MIN_WIDTH", 1280)
MIN_HEIGHT = _i("SOURCE_MIN_HEIGHT", 720)
MIN_FPS = _f("SOURCE_MIN_FPS", 24.0)
MIN_VIDEO_MB = _f("SOURCE_MIN_VIDEO_MB", 5.0)
MIN_IMAGES = _i("SOURCE_MIN_IMAGES", 8)
SAMPLE_FRAMES = _i("SOURCE_SAMPLE_FRAMES", 12)
MIN_SHARP_PASS_RATE = _f("SOURCE_MIN_SHARP_PASS_RATE", 0.6)
MIN_BLUR_SCORE = _f("SOURCE_MIN_BLUR_SCORE", 12.0)
MIN_BRIGHTNESS = _f("SOURCE_MIN_BRIGHTNESS", 16.0)
MAX_BRIGHTNESS = _f("SOURCE_MAX_BRIGHTNESS", 240.0)

_GUIDANCE = (
    "Supply a longer walk-around video (front, both sides, rear with "
    "overlap, good light) or add more reference images of the missing "
    "sides before re-running processing."
)


def _ffprobe_meta(video: Path, ffprobe: str) -> Dict[str, Any]:
    command = [
        ffprobe, "-v", "error",
        "-show_streams", "-show_format", "-of", "json", str(video),
    ]
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"ffprobe could not read the video: {exc}") from exc
    if proc.returncode != 0:
        raise ValueError("ffprobe rejected the video file")
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe returned invalid metadata") from exc
    streams = payload.get("streams") or []
    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    if not video_stream:
        raise ValueError("no video stream found")
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    fps = _parse_rate(
        video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
    )
    try:
        duration = float(payload.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    frame_count = int(video_stream.get("nb_frames") or 0)
    if frame_count <= 0 and fps > 0 and duration > 0:
        frame_count = int(fps * duration)
    return {
        "codec": video_stream.get("codec_name"),
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": duration,
        "frame_count": frame_count,
        "size_mb": video.stat().st_size / (1024 * 1024),
        "command": " ".join(shlex.quote(p) for p in command),
    }


def _parse_rate(value: Any) -> float:
    try:
        if not value:
            return 0.0
        if "/" in str(value):
            num, _, den = str(value).partition("/")
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _sample_frames(video: Path, count: int) -> List[Dict[str, Any]]:
    """Evenly spaced frame sharpness/brightness via ffmpeg + OpenCV.

    Returns per-sample dicts; empty list when sampling is unavailable
    (no ffmpeg/cv2) so the gate degrades to metadata-only checks.
    """
    try:
        import cv2  # noqa: PLC0415  (optional: metadata-only without it)
    except Exception:
        return []
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    import tempfile

    results: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="modz-source-check-") as tmp:
        pattern = str(Path(tmp) / "sample_%03d.jpg")
        # One thumbnail per second, capped at `count`: evenly spread samples
        # across the clip without decoding every frame.
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-vf", f"fps=1",
            "-frames:v", str(count), pattern,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=300, check=False)
        except (OSError, subprocess.SubprocessError):
            return []
        for thumb in sorted(Path(tmp).glob("sample_*.jpg")):
            img = cv2.imread(str(thumb), cv2.IMREAD_GRAYSCALE)
            if img is None:
                results.append({"readable": False})
                continue
            blur = float(cv2.Laplacian(img, cv2.CV_64F).var())
            brightness = float(img.mean())
            results.append({
                "readable": True,
                "blur_score": round(blur, 1),
                "brightness": round(brightness, 1),
                "sharp": blur >= MIN_BLUR_SCORE,
                "exposed": MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS,
            })
    return results


def _resolution_verdict(width: int, height: int) -> Tuple[bool, str, Optional[str]]:
    """Orientation-aware resolution gate.

    Full marks for 720p-or-better in either orientation.  Smaller footage
    still passes if it carries enough pixels for SIFT (long side >= 800,
    >= ~300k pixels - roughly 640x480) but gets a low-detail warning;
    anything below that cannot feed feature matching and is blocked.
    """
    full = (
        (width >= MIN_WIDTH and height >= MIN_HEIGHT)
        or (height >= MIN_WIDTH and width >= MIN_HEIGHT)
    )
    detail = f"{width}x{height} (recommended {MIN_WIDTH}x{MIN_HEIGHT} or portrait equivalent)"
    if full:
        return True, detail, None
    long_side = max(width, height)
    if long_side >= 800 and width * height >= 300_000:
        return True, detail, (
            "below 720p detail - expect fewer features and check the "
            "3D Preview for missing sides"
        )
    return False, detail + " - too small for feature matching", None


def check_video_source(
    video: Path, *, ffprobe: Optional[str] = None
) -> Dict[str, Any]:
    """Verify one reference video.  Never raises for bad input: reports it."""
    checks: List[Dict[str, Any]] = []
    failures: List[str] = []
    warnings: List[str] = []

    def add(name: str, ok: bool, detail: str, *, blocking: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            (failures if blocking else warnings).append(f"{name}: {detail}")

    if not video.is_file() or video.stat().st_size == 0:
        add("file present", False, "video file is missing or empty")
        return _report(False, checks, failures, warnings, {})
    tool = ffprobe or shutil.which("ffprobe") or "ffprobe"
    try:
        meta = _ffprobe_meta(video, tool)
    except ValueError as exc:
        add("readable video stream", False, str(exc))
        return _report(False, checks, failures, warnings, {})
    add("readable video stream", True,
        f"{meta['codec'] or 'unknown codec'}, {meta['width']}x{meta['height']}")
    size_ok = meta["size_mb"] >= MIN_VIDEO_MB
    add("file size", size_ok,
        f"{meta['size_mb']:.1f} MB (minimum {MIN_VIDEO_MB:.0f} MB)",
        blocking=False)
    dur_ok = meta["duration_seconds"] >= MIN_DURATION_S
    add("duration", dur_ok,
        f"{meta['duration_seconds']:.1f}s (minimum {MIN_DURATION_S:.0f}s) - "
        f"short clips cannot cover the full orbit")
    res_ok, res_detail, res_warn = _resolution_verdict(
        meta["width"], meta["height"])
    add("resolution", res_ok, res_detail)
    if res_warn:
        warnings.append(f"resolution: {res_warn}")
    fps_ok = meta["fps"] >= MIN_FPS or meta["fps"] <= 0
    # fps <= 0 means the container hides the rate; warn instead of fail.
    if meta["fps"] <= 0:
        add("frame rate", True,
            "frame rate not reported by container - assuming camera default",
            blocking=False)
    else:
        add("frame rate", fps_ok,
            f"{meta['fps']:.1f} fps (minimum {MIN_FPS:.0f} fps)")
    samples = _sample_frames(video, SAMPLE_FRAMES)
    usable = [s for s in samples if s.get("readable")]
    if samples and usable:
        sharp = sum(1 for s in usable if s["sharp"] and s["exposed"])
        rate = sharp / max(1, len(usable))
        add("frame quality", rate >= MIN_SHARP_PASS_RATE,
            f"{sharp}/{len(usable)} sampled frames sharp and well-exposed "
            f"(need >= {int(MIN_SHARP_PASS_RATE * 100)}%) - blurry/dark "
            f"footage starves feature matching")
    elif samples:
        add("frame quality", False, "sampled frames could not be decoded")
    else:
        warnings.append("frame quality: sampler unavailable - metadata checks only")
        checks.append({"name": "frame quality", "ok": True,
                       "detail": "sampler unavailable - metadata checks only"})
    ok = not failures
    return _report(ok, checks, failures, warnings, meta)


def check_image_set(paths: List[Path]) -> Dict[str, Any]:
    """Verify a set of reference photographs."""
    checks: List[Dict[str, Any]] = []
    failures: List[str] = []
    warnings: List[str] = []
    readable = [p for p in paths if p.is_file() and p.stat().st_size > 0]
    ok_count = len(readable) >= MIN_IMAGES
    checks.append({"name": "image count", "ok": ok_count,
                   "detail": f"{len(readable)} readable images "
                             f"(minimum {MIN_IMAGES} for an orbit)"})
    if not ok_count:
        failures.append(
            f"image count: only {len(readable)} usable images - "
            f"add images of the missing sides")
    small = 0
    try:
        from PIL import Image  # noqa: PLC0415 (optional)
        for path in readable:
            try:
                with Image.open(path) as img:
                    if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
                        small += 1
            except Exception:
                small += 1
        if small:
            detail = (f"{small}/{len(readable)} images below "
                      f"{MIN_WIDTH}x{MIN_HEIGHT}")
            if small > len(readable) // 2:
                checks.append({"name": "image resolution", "ok": False, "detail": detail})
                failures.append(f"image resolution: {detail}")
            else:
                checks.append({"name": "image resolution", "ok": True, "detail": detail})
                warnings.append(f"image resolution: {detail}")
        else:
            checks.append({"name": "image resolution", "ok": True,
                           "detail": f"all {len(readable)} meet {MIN_WIDTH}x{MIN_HEIGHT}"})
    except ImportError:
        checks.append({"name": "image resolution", "ok": True,
                       "detail": "Pillow unavailable - count check only"})
    return _report(not failures, checks, failures, warnings,
                   {"count": len(readable)})


def _report(
    ok: bool,
    checks: List[Dict[str, Any]],
    failures: List[str],
    warnings: List[str],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "ok": ok,
        "score": f"{sum(1 for c in checks if c['ok'])}/{len(checks)}",
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "guidance": None if ok else _GUIDANCE,
        "meta": meta,
        "thresholds": {
            "min_duration_s": MIN_DURATION_S,
            "min_width": MIN_WIDTH,
            "min_height": MIN_HEIGHT,
            "min_fps": MIN_FPS,
            "min_video_mb": MIN_VIDEO_MB,
            "min_images": MIN_IMAGES,
        },
    }
