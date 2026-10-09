"""Background masking between SfM and dense reconstruction.

Why this stage exists: the engine reconstructs whatever has consistent
features.  A small dark object on a bright table loses to the background -
the table covers more pixels, so most SIFT matches, sparse points, and mesh
chunks describe the table, not the object.

Why the stage sits *after* SfM, not before it: camera poses need those
background features.  A masked frame is mostly black, and black has no
SIFT features - masking first starves pose estimation (measured: 32/35
frames registered unmasked vs 2/35 masked).  So features, matching, and
SfM run on the full frames, and only then is the background painted
solid black for the geometry stages (dense, mesh, texture), which read
``frames/masked`` while pose stages keep reading ``frames/selected``.

Two backends, no new hard dependency:

* ``rembg`` (u2netp, ~5 MB, CPU) when installed - real foreground
  segmentation that works on any background.  Optional: ``pip install
  rembg onnxruntime``.
* Bright-background keying (always available, cv2 only): when the frame
  border is uniformly bright the background is thresholded away and only
  the largest contour is kept.  A strict guard skips frames whose border
  is not uniform-bright, because a wrong mask is worse than no mask.

Masked pixels become pure black.  Dense matching, TSDF fusion, and texture
baking all treat black as "no information", which is exactly right.

``frames/selected`` is never modified - masked copies live in
``frames/masked``.  Toggling the mask flags invalidates the dense stage
and everything downstream via the dense fingerprint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config import ReconstructionError


def _cv2():
    import cv2
    return cv2


def _rembg_mask(image_bgr: "np.ndarray", model: str = "u2netp") -> Optional["np.ndarray"]:
    """Foreground mask via rembg when installed, else None.

    The image is downscaled to 640 px on the long side first: u2net on a
    full 1080p frame takes minutes on this CPU, and mask edges do not
    need full resolution (they are feathered on composite anyway).
    """
    try:
        from rembg import remove, new_session
    except Exception:
        return None
    try:
        from PIL import Image
        import io
        cv2 = _cv2()
        height, width = image_bgr.shape[:2]
        scale = 640.0 / max(height, width)
        small = image_bgr if scale >= 1.0 else cv2.resize(
            image_bgr, (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", small)
        if not ok:
            return None
        session = new_session(model)
        cutout = remove(Image.open(io.BytesIO(encoded.tobytes())),
                        session=session)
        alpha = np.array(cutout.split()[-1])
        if scale < 1.0:
            alpha = cv2.resize(alpha, (width, height),
                               interpolation=cv2.INTER_LINEAR)
        return (alpha > 127).astype(np.uint8) * 255
    except Exception:
        return None


def _keying_mask(gray: "np.ndarray") -> Optional["np.ndarray"]:
    """Mask a uniformly bright background; None when the guard rejects."""
    cv2 = _cv2()
    height, width = gray.shape
    margin = max(4, min(height, width) // 40)
    border = np.concatenate([
        gray[:margin, :].ravel(),
        gray[-margin:, :].ravel(),
        gray[:, :margin].ravel(),
        gray[:, -margin:].ravel(),
    ]).astype(np.float32)
    bg_mean = float(border.mean())
    bg_std = float(border.std())
    # Guard: only a bright, even background is safe to key away.
    if bg_mean < 195 or bg_std > 30:
        return None
    _, foreground = cv2.threshold(
        gray, max(150.0, bg_mean - 35.0), 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, margin // 2) * 2 + 1,) * 2)
    cleaned = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    coverage = float(cv2.contourArea(largest)) / (height * width)
    # An object that fills (almost) nothing or everything is a keying
    # failure, not a subject - leave the frame alone.
    if coverage < 0.02 or coverage > 0.92:
        return None
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
    return mask


def _mask_frame(source: Path, destination: Path, backend: str,
                model: str = "u2netp") -> str:
    """Write a background-masked copy.  Returns 'rembg', 'keying', or 'skipped'."""
    cv2 = _cv2()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        return "skipped"
    mask: Optional["np.ndarray"] = None
    used = "skipped"
    if backend in {"auto", "rembg"}:
        mask = _rembg_mask(image, model)
        if mask is not None:
            used = "rembg"
    if mask is None and backend in {"auto", "keying"}:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = _keying_mask(gray)
        if mask is not None:
            used = "keying"
    if mask is None:
        return "skipped"
    soft = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
    composed = (image.astype(np.float32) * soft[..., None]).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), composed,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return used


def run_masking(cfg, store, ctx) -> Dict[str, Any]:
    """Copy selected frames to frames/masked with backgrounds blacked out."""
    if not bool(getattr(cfg, "mask_background", True)):
        # Clear stale masks: _frames_by_name prefers frames/masked whenever
        # files exist there, so a disabled run must not inherit them.
        masked_dir = cfg.d("frames", "masked")
        if masked_dir.is_dir():
            for stale in masked_dir.glob("frame_*.jpg"):
                try:
                    stale.unlink()
                except OSError:
                    pass
        result = {"enabled": False, "masked": 0, "skipped": 0,
                  "backend": "off"}
        ctx.result(result)
        return result
    selected_dir = cfg.d("frames", "selected")
    masked_dir = cfg.d("frames", "masked")
    frames = sorted(selected_dir.glob("frame_*.jpg"))
    if not frames:
        raise ReconstructionError(
            "Background masking found no selected frames.",
            suggestion="Run frame selection first (stage order handles this).",
        )
    # Fresh output: stale masks from a previous configuration must not leak
    # into the geometry stages.
    if masked_dir.is_dir():
        for stale in masked_dir.glob("frame_*.jpg"):
            try:
                stale.unlink()
            except OSError:
                pass
    else:
        masked_dir.mkdir(parents=True, exist_ok=True)
    backend = str(getattr(cfg, "mask_backend", "auto")).lower()
    if backend not in {"auto", "rembg", "keying"}:
        backend = "auto"
    model = str(getattr(cfg, "mask_model", "u2netp")).strip() or "u2netp"
    masked = skipped = 0
    rembg_used = keying_used = 0
    for path in frames:
        ctx.check()
        outcome = _mask_frame(path, masked_dir / path.name, backend, model)
        if outcome == "skipped":
            skipped += 1
        else:
            masked += 1
            if outcome == "rembg":
                rembg_used += 1
            else:
                keying_used += 1
    result = {
        "enabled": True,
        "backend": backend,
        "directory": "frames/masked",
        "masked": masked,
        "skipped": skipped,
        "rembg_frames": rembg_used,
        "keying_frames": keying_used,
        "total": len(frames),
        "note": ("Frames left untouched (busy background, guard refused) "
                 "reconstruct with their background as before.")
        if skipped else "All selected frames masked.",
    }
    (cfg.d("frames") / "masking.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    ctx.result(result)
    return result
