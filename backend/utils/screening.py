"""Automatic photogrammetry screening of a reference video.

``utils.source_check`` asks "is this a usable video file" (duration,
resolution, sharpness).  This module asks the question that actually
predicts whether a *3D model* comes out: does the camera move around the
subject, is the subject textured, and is the clip one continuous take
rather than a montage of unrelated shots?

Everything is measured on a handful of evenly spaced thumbnails decoded
at working size, so screening a 3-minute clip costs a few seconds:

* **camera travel** - Farneback optical flow between consecutive samples,
  median displacement as a percentage of frame width.  A real orbit
  walks the camera several frame-widths across the subject; a tripod
  pan in place, a zoom, or a locked-off shot gives almost none, and
  without parallax there is no depth to reconstruct.
* **cuts** - ORB match count *and* histogram correlation between
  consecutive samples (both must fire: measured on real clips, either
  signal alone misfires - pairs of genuinely consecutive views of a
  compressed YouTube clip match with 0 good ORB points, while a slow
  orbit legitimately drops histogram correlation as the background
  leaves frame).  A montage breaks the sequential matcher: frames from
  different shots never co-register, so whole sides go missing.
* **texture** - ORB keypoints per sample.  Textureless subjects (matte
  black, blank plastic, plain paint) give SIFT nothing to match, which
  is the single most common reason a real object reconstructs as
  fragments.
* **exposure stability** - brightness swing across the clip; heavy
  flicker desynchronises matching between frames.

The report uses the same shape as ``utils.source_check`` (ok / score /
checks / failures / warnings / guidance) so the pipeline gate and the UI
render both identically.  Screening never raises: with no ffmpeg or no
OpenCV it degrades to a warning and lets the metadata checks decide.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

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


SAMPLE_COUNT = _i("SCREEN_SAMPLE_COUNT", 16)
WORKING_WIDTH = _i("SCREEN_WORKING_WIDTH", 640)
MIN_TRAVEL_PCT = _f("SCREEN_MIN_TRAVEL_PCT", 0.25)    # parallax floor
GOOD_TRAVEL_PCT = _f("SCREEN_GOOD_TRAVEL_PCT", 4.0)   # comfortable orbit
MAX_TRAVEL_PCT = _f("SCREEN_MAX_TRAVEL_PCT", 30.0)    # whip-pan ceiling
CUT_CORR = _f("SCREEN_CUT_CORR", 0.6)                 # cut histogram gate
MAX_CUT_SHARE = _f("SCREEN_MAX_CUT_SHARE", 0.25)
MIN_TEXTURE_KP = _i("SCREEN_MIN_TEXTURE_KP", 150)
GOOD_TEXTURE_KP = _i("SCREEN_GOOD_TEXTURE_KP", 400)
MAX_BRIGHTNESS_SWING = _f("SCREEN_MAX_BRIGHTNESS_SWING", 45.0)

_GUIDANCE = (
    "Re-shoot with a slow steady orbit 1-2 m from the subject, keeping "
    "it filling the frame, even light and no cuts; if the surface is "
    "smooth or dark, add texture (matte spray, stickers, flour dusting) "
    "before filming."
)


def _probe_duration(video: Path, ffprobe: str) -> float:
    command = [
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(video),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              timeout=120)
        payload = json.loads(proc.stdout or "{}")
        return float(payload.get("format", {}).get("duration") or 0.0)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return 0.0


def _sample(video: Path, count: int, duration_s: float) -> List[Path]:
    """Evenly spaced thumbnails at working size (grayscale JPEGs)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    tmp = Path(tempfile.mkdtemp(prefix="modz-screen-"))
    rate = max(0.05, count / duration_s) if duration_s > 0 else 1.0
    pattern = str(tmp / "s_%03d.jpg")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-vf", f"fps={rate:.4f},scale={WORKING_WIDTH}:-2",
        "-frames:v", str(count), "-pix_fmt", "gray", pattern,
    ]
    try:
        subprocess.run(command, capture_output=True, timeout=600, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted(tmp.glob("s_*.jpg"))


def _metrics(frames: List[Path]) -> Dict[str, Any]:
    """Flow travel, cut share, texture, exposure swing over the samples."""
    import cv2  # noqa: PLC0415  (optional dependency)

    images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in frames]
    images = [img for img in images if img is not None]
    if len(images) < 3:
        raise ValueError("too few readable samples")

    width = images[0].shape[1]
    orb = cv2.ORB_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    travels: List[float] = []
    cuts = 0
    pairs = 0
    keypoints: List[int] = []
    brightness: List[float] = []
    descriptors = [orb.detectAndCompute(img, None) for img in images]

    # Per-pair feature chaining plus histogram correlation.  Either
    # signal alone misfires on real footage (measured: pairs of two
    # genuinely consecutive views of a YouTube clip match with 0 good ORB
    # points because of compression, while a slow orbit legitimately
    # drops histogram correlation as the background leaves frame).  Both
    # firing together is what a scene cut looks like.
    good_counts: List[int] = []
    correlations: List[float] = []
    for index in range(1, len(images)):
        _kp_prev, des_prev = descriptors[index - 1]
        _kp_now, des_now = descriptors[index]
        good = 0
        if des_prev is not None and des_now is not None and len(des_prev) >= 2:
            try:
                knn = matcher.knnMatch(des_prev, des_now, k=2)
                good = sum(1 for m in knn
                           if len(m) == 2 and m[0].distance < 0.8 * m[1].distance)
            except cv2.error:
                good = 0
        hist_prev = cv2.calcHist([images[index - 1]], [0], None, [64], [0, 256])
        hist_now = cv2.calcHist([images[index]], [0], None, [64], [0, 256])
        cv2.normalize(hist_prev, hist_prev)
        cv2.normalize(hist_now, hist_now)
        good_counts.append(good)
        correlations.append(float(cv2.compareHist(hist_prev, hist_now,
                                                  cv2.HISTCMP_CORREL)))

    median_good = sorted(good_counts)[len(good_counts) // 2]
    cut_floor = max(8, 0.3 * median_good)
    for index in range(1, len(images)):
        pairs += 1
        is_cut = (good_counts[index - 1] < cut_floor
                  and correlations[index - 1] < CUT_CORR)
        if is_cut:
            cuts += 1
            continue  # flow across a cut is meaningless
        flow = cv2.calcOpticalFlowFarneback(
            images[index - 1], images[index], None, pyr_scale=0.5, levels=3,
            winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        travels.append(float(cv2.medianBlur(magnitude, 5).mean()
                             / max(1, width) * 100.0))

    for index, img in enumerate(images):
        keypoints.append(len(descriptors[index][0]))
        brightness.append(float(img.mean()))

    if not travels:
        raise ValueError("every consecutive pair looks like a cut")
    travels.sort()
    return {
        "samples": len(images),
        "travel_pct_median": round(travels[len(travels) // 2], 2),
        "travel_pct_max": round(travels[-1], 2),
        "cut_share": round(cuts / max(1, pairs), 3),
        "cuts": cuts,
        "chaining_matches_median": int(median_good),
        "keypoints_median": int(sorted(keypoints)[len(keypoints) // 2]),
        "brightness_swing": round(max(brightness) - min(brightness), 1),
    }

    if not travels:
        raise ValueError("every consecutive pair looks like a cut")
    travels.sort()
    return {
        "samples": len(images),
        "travel_pct_median": round(travels[len(travels) // 2], 2),
        "travel_pct_max": round(travels[-1], 2),
        "cut_share": round(cuts / max(1, pairs), 3),
        "cuts": cuts,
        "keypoints_median": int(sorted(keypoints)[len(keypoints) // 2]),
        "brightness_swing": round(max(brightness) - min(brightness), 1),
    }


def screen_video(video: Path, *, ffprobe: Optional[str] = None
                 ) -> Dict[str, Any]:
    """Judge a clip for reconstructability.  Never raises."""
    checks: List[Dict[str, Any]] = []
    failures: List[str] = []
    warnings: List[str] = []

    def add(name: str, ok: bool, detail: str, *, blocking: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            (failures if blocking else warnings).append(f"{name}: {detail}")

    ffprobe = ffprobe or shutil.which("ffprobe") or "ffprobe"
    duration = _probe_duration(video, ffprobe)
    frames = _sample(video, SAMPLE_COUNT, duration)
    if not frames:
        return {
            "ok": True, "score": "n/a", "checks": [], "failures": [],
            "warnings": ["footage screening skipped: ffmpeg unavailable"],
            "guidance": None, "meta": {},
        }
    try:
        meta = _metrics(frames)
    except Exception as exc:  # cv2 missing, unreadable frames, all cuts
        return {
            "ok": True, "score": "n/a", "checks": [], "failures": [],
            "warnings": [f"footage screening skipped: {exc}"],
            "guidance": None, "meta": {},
        }

    travel = meta["travel_pct_median"]
    if travel < MIN_TRAVEL_PCT:
        add("camera travel", False,
            f"only {travel}% of frame width between views - the camera "
            f"barely moves around the subject, so there is no parallax to "
            f"reconstruct")
    elif travel < GOOD_TRAVEL_PCT:
        add("camera travel", True,
            f"{travel}% per view - workable but a slower, wider orbit "
            f"would fill in more sides", blocking=False)
    elif travel > MAX_TRAVEL_PCT:
        add("camera travel", False,
            f"{travel}% per view - the camera whips between views; "
            f"matching will lose the subject", blocking=False)
    else:
        add("camera travel", True, f"{travel}% per view - good parallax")

    cut_share = meta["cut_share"]
    if cut_share > MAX_CUT_SHARE:
        # Advisory only: on compressed footage a cut and a hard-to-match
        # but continuous view are not separable at sampling resolution,
        # and refusing footage that would have reconstructed wastes far
        # more of the user's time than letting the matcher try.
        add("single continuous take", False,
            f"{meta['cuts']} of {meta['samples']} views may be scene "
            f"cuts - if the model comes back with missing sides, "
            f"re-shoot as one unbroken orbit", blocking=False)
    elif cut_share > 0:
        add("single continuous take", True,
            f"{meta['cuts']} possible cut(s) of {meta['samples']} views - "
            f"mostly continuous", blocking=False)
    else:
        add("single continuous take", True, "no cuts detected")

    texture = meta["keypoints_median"]
    if texture < MIN_TEXTURE_KP:
        add("surface texture", False,
            f"~{texture} feature points per view - the surface is too "
            f"smooth/textureless for matching")
    elif texture < GOOD_TEXTURE_KP:
        add("surface texture", True,
            f"~{texture} feature points per view - thin but usable",
            blocking=False)
    else:
        add("surface texture", True, f"~{texture} feature points per view")

    swing = meta["brightness_swing"]
    if swing > MAX_BRIGHTNESS_SWING:
        add("stable exposure", False,
            f"brightness swings {swing} levels across the clip",
            blocking=False)
    else:
        add("stable exposure", True, f"brightness swing {swing} levels")

    passed = sum(1 for c in checks if c["ok"])
    return {
        "ok": not failures,
        "score": f"{passed}/{len(checks)}",
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "guidance": None if not failures else _GUIDANCE,
        "meta": meta,
    }


def render_command(video: Path, ffprobe: Optional[str] = None) -> str:
    """Human-readable command line for the stage log."""
    ffprobe = ffprobe or shutil.which("ffprobe") or "ffprobe"
    return " ".join(shlex.quote(p) for p in (
        ffprobe, "-show_entries", "format=duration", str(video),
        "+ ffmpeg fps/scale sampling + Farneback flow/ORB screening"))
