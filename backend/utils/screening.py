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
* **continuous camera motion** - the share of consecutive sample pairs
  whose true median flow is effectively zero (a still frame).  This is
  the check that catches a *showcase reel*: a montage of still renders
  joined by crossfades, where each pair either holds a frozen image or
  dissolves between two unrelated views.  The mean flow gets lifted by
  the fading edges and reports motion that is not there (measured: a
  matatu showcase clip showed 0.69% mean travel - clearing the gate -
  but a 0.00px true median and 85% of pairs completely static).  The
  reconstruction that follows locks onto burned-in overlays and the
  flat backdrop, producing a mesh that bears no resemblance to the
  subject on screen.
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
* **texture belongs to the subject** - how many of the frame's features
  actually move with the scene.  A burned-in logo/watermark or a static
  backdrop contributes features that never move; matching locks onto
  them (they are the only perfectly visible, noise-free points) and the
  model becomes a slab at the overlay plane instead of the subject.
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
MIN_TRAVEL_PCT = _f("SCREEN_MIN_TRAVEL_PCT", 0.05)    # parallax floor
GOOD_TRAVEL_PCT = _f("SCREEN_GOOD_TRAVEL_PCT", 0.8)   # comfortable orbit
MAX_TRAVEL_PCT = _f("SCREEN_MAX_TRAVEL_PCT", 6.0)     # whip-pan ceiling
# A pair whose true median flow sits under this is a still frame, not a
# view.  0.1% of a 640px frame is 0.64px: measured real orbit footage
# runs 0.8-1.0%, a slideshow of stills with crossfades 0.00%.
STATIC_TRAVEL_PCT = _f("SCREEN_STATIC_TRAVEL_PCT", 0.1)
MAX_STATIC_PAIR_SHARE = _f("SCREEN_MAX_STATIC_PAIR_SHARE", 0.5)
CUT_CORR = _f("SCREEN_CUT_CORR", 0.6)                 # cut histogram gate
MAX_CUT_SHARE = _f("SCREEN_MAX_CUT_SHARE", 0.25)
FROZEN_FLOW_PX = _f("SCREEN_FROZEN_FLOW_PX", 1.0)     # overlay = no flow
MAX_FROZEN_SHARE = _f("SCREEN_MAX_FROZEN_SHARE", 0.5)  # texture in overlay
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


def _static_texture(images: List["Any"], orb: Any
                    ) -> Dict[str, Any]:
    """How much of the frame's texture never moves.

    A burned-in overlay (studio logo, watermark, lower-third text) sits
    in the same pixels in every frame while the subject rotates behind
    it.  Feature matching then locks onto the overlay - it is the only
    thing with perfect, noise-free, constantly-visible features - and the
    reconstruction becomes a slab at the overlay plane instead of the
    subject.  Measured on a matatu showcase clip: 99/102 views
    registered, but only 3269 sparse points, all on the watermark.

    So texture only counts when it moves with the scene.  A pixel is
    "frozen" when its optical flow stays below a pixel across every
    consecutive pair while the frame overall moves; texture that lives
    almost entirely in frozen pixels is overlay, not subject.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    frozen_flow = np.zeros(images[0].shape, dtype=np.float32)
    pairs = 0
    for index in range(1, len(images)):
        flow = cv2.calcOpticalFlowFarneback(
            images[index - 1], images[index], None, pyr_scale=0.5, levels=3,
            winsize=21, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        frozen_flow = np.maximum(frozen_flow,
                                 cv2.magnitude(flow[..., 0], flow[..., 1]))
        pairs += 1
    frozen = frozen_flow < FROZEN_FLOW_PX

    keypoints = orb.detect(images[len(images) // 2], None)
    if not keypoints:
        return {"frozen_keypoint_share": 0.0, "moving_keypoints": 0,
                "frozen_keypoints": 0}
    frozen_count = sum(1 for kp in keypoints
                       if frozen[int(kp.pt[1]), int(kp.pt[0])])
    return {
        "frozen_keypoint_share": round(frozen_count / len(keypoints), 3),
        "moving_keypoints": int(len(keypoints) - frozen_count),
        "frozen_keypoints": int(frozen_count),
        "frozen_pixel_share": round(float(frozen.mean()), 3),
    }


def _metrics(frames: List[Path]) -> Dict[str, Any]:
    """Flow travel, cut share, texture, exposure swing over the samples."""
    import cv2  # noqa: PLC0415  (optional dependency)
    import numpy as np  # noqa: PLC0415

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
        # True median, never the mean of a median-blurred field.  A
        # slideshow of still renders joined by crossfades moves only the
        # fading edges while the whole frame is frozen; the mean is
        # lifted by exactly that small region and reports motion that is
        # not there.  Measured on a showcase clip: mean-of-median-blur
        # 0.69% (cleared the gate) while the true median was 0.00px and
        # 74% of pairs were completely static.
        travels.append(float(np.median(magnitude) / max(1, width) * 100.0))

    for index, img in enumerate(images):
        keypoints.append(len(descriptors[index][0]))
        brightness.append(float(img.mean()))

    overlay = _static_texture(images, orb)

    if not travels:
        raise ValueError("every consecutive pair looks like a cut")
    static_share = (sum(1 for value in travels if value < STATIC_TRAVEL_PCT)
                    / len(travels))
    travels.sort()
    return {
        "samples": len(images),
        "travel_pct_median": round(travels[len(travels) // 2], 2),
        "travel_pct_max": round(travels[-1], 2),
        "static_pair_share": round(static_share, 3),
        "cut_share": round(cuts / max(1, pairs), 3),
        "cuts": cuts,
        "chaining_matches_median": int(median_good),
        "keypoints_median": int(sorted(keypoints)[len(keypoints) // 2]),
        "brightness_swing": round(max(brightness) - min(brightness), 1),
        **overlay,
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

    # A slideshow of stills joined by crossfades: most pairs show no
    # motion at all, only the fade itself.  Photogrammetry needs a
    # camera walking around one subject; a montage of separate renders
    # cannot triangulate.  Measured on a matatu showcase clip: 74% of
    # pairs fully static, 3269 sparse points, mesh bore no resemblance
    # to the vehicle on screen.
    static_share = meta.get("static_pair_share", 0.0)
    if static_share > MAX_STATIC_PAIR_SHARE:
        add("continuous camera motion", False,
            f"{int(static_share * 100)}% of consecutive views show no "
            f"motion - this is a slideshow/cutshow of still shots, not a "
            f"continuous orbit around one subject")
    else:
        add("continuous camera motion", True,
            f"{int(static_share * 100)}% of consecutive views are still")

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

    # Texture that never moves is a burned-in overlay, not the subject.
    # Matching locks onto it and the model comes out as a slab at the
    # overlay plane (measured: a watermark-only showcase clip gave
    # 3269 sparse points and a mesh that bore no resemblance to the
    # vehicle in the video).
    frozen_share = meta.get("frozen_keypoint_share", 0.0)
    moving_kp = meta.get("moving_keypoints", 0)
    if frozen_share > MAX_FROZEN_SHARE and moving_kp < GOOD_TEXTURE_KP:
        add("texture belongs to the subject", False,
            f"{int(frozen_share * 100)}% of features never move while the "
            f"scene does - that is a burned-in logo/watermark or a static "
            f"backdrop, and only ~{moving_kp} moving features remain; the "
            f"reconstruction would lock onto the overlay, not the subject")
    elif frozen_share > MAX_FROZEN_SHARE:
        add("texture belongs to the subject", False,
            f"{int(frozen_share * 100)}% of features never move - a "
            f"burned-in overlay dominates the frame", blocking=False)
    else:
        add("texture belongs to the subject", True,
            f"{int((1 - frozen_share) * 100)}% of features move with the "
            f"scene")

    swing = meta["brightness_swing"]
    if swing > MAX_BRIGHTNESS_SWING:
        add("stable exposure", False,
            f"brightness swings {swing} levels across the clip",
            blocking=False)
    else:
        add("stable exposure", True, f"brightness swing {swing} levels")

    passed = sum(1 for c in checks if c["ok"])
    # Advice that names the actual problem.  A showcase reel (still
    # renders joined by crossfades) is not a slow orbit, so telling the
    # user to "re-shoot with a slower orbit" misdiagnoses it - the
    # footage is not of the real object at all and no setting recovers a
    # model from it.
    guidance: Optional[str] = None
    if failures:
        if meta.get("static_pair_share", 0.0) > MAX_STATIC_PAIR_SHARE:
            guidance = (
                "This clip is a montage of still shots (a showcase/reel), "
                "not one continuous orbit around a single object, so the "
                "camera positions cannot be triangulated. Film the real "
                "subject yourself: walk one slow unbroken 360-degree orbit "
                "1-2 m away, keeping it filling the frame under even light, "
                "and avoid cuts, zooms, or edits between separate shots."
            )
        elif meta.get("keypoints_median", 0) < MIN_TEXTURE_KP:
            guidance = (
                "The surface is too smooth or dark for the matcher to "
                "find features. Add texture before filming (matte spray, "
                "stickers, or a dusting of flour), shoot under diffuse "
                "even light, and keep the camera 1-2 m away so the "
                "surface fills the frame with visible detail."
            )
        else:
            guidance = _GUIDANCE
    return {
        "ok": not failures,
        "score": f"{passed}/{len(checks)}",
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "guidance": guidance,
        "meta": meta,
    }


def render_command(video: Path, ffprobe: Optional[str] = None) -> str:
    """Human-readable command line for the stage log."""
    ffprobe = ffprobe or shutil.which("ffprobe") or "ffprobe"
    return " ".join(shlex.quote(p) for p in (
        ffprobe, "-show_entries", "format=duration", str(video),
        "+ ffmpeg fps/scale sampling + Farneback flow/ORB screening"))
