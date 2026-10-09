"""PHASE 2a - SIFT feature detection and description (CPU only).

One image is loaded, described, written to disk, and released before the
next one starts: the E6410 never holds the whole frame set in RAM.  Each
image gets ``features/<name>.npz`` so the stage is resumable - a restart
re-uses every descriptor already on disk.

SIFT is the default (section 7): mature, CPU-native, repeatable across
scale and viewpoint, and no neural network is involved.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import ReconConfig, ReconstructionError
from .state import StateStore, StageContext, memory_pressure


def _cv2():
    import cv2
    return cv2


def selected_frames(cfg: ReconConfig) -> List[Path]:
    selected = cfg.d("frames", "selected")
    frames = sorted(selected.glob("frame_*.jpg"))
    if frames:
        return frames
    # Legacy layout from the web backend: frames/frame_*.jpg
    frames = sorted(cfg.d("frames").glob("frame_*.jpg"))
    return frames


def _npz_path(cfg: ReconConfig, frame: Path) -> Path:
    return cfg.d("features", frame.stem + ".npz")


def load_features(cfg: ReconConfig, frame: Path) -> Optional[Dict[str, Any]]:
    path = _npz_path(cfg, frame)
    if not path.is_file():
        return None
    try:
        import numpy as np
        data = np.load(path, allow_pickle=False)
        return {
            "kp": data["kp"],            # (N,5): x, y, size, angle, response
            "desc": data["desc"],        # (N,128) uint8
            "width": int(data["width"]),
            "height": int(data["height"]),
            "orig_width": int(data["orig_width"]),
            "orig_height": int(data["orig_height"]),
        }
    except Exception:
        # Truncated/corrupt file (interrupted run) -> recompute it.
        return None


def load_keypoints_only(cfg: ReconConfig, frame: Path) -> Optional[Dict[str, Any]]:
    """Memory-cheap load for SfM: positions and sizes, no descriptors."""
    path = _npz_path(cfg, frame)
    if not path.is_file():
        return None
    try:
        import numpy as np
        data = np.load(path, allow_pickle=False)
        return {
            "kp": data["kp"],
            "width": int(data["width"]),
            "height": int(data["height"]),
            "orig_width": int(data["orig_width"]),
            "orig_height": int(data["orig_height"]),
        }
    except Exception:
        return None


def extract_features(cfg: ReconConfig, store: StateStore, ctx: StageContext) -> Dict[str, Any]:
    cv2 = _cv2()
    frames = selected_frames(cfg)
    if not frames:
        raise ReconstructionError(
            "No selected frames found for feature extraction.",
            suggestion="Run the frame selection stage first; frames must exist "
                       "in frames/selected/ (or legacy frames/).",
        )
    cfg.d("features").mkdir(parents=True, exist_ok=True)
    meta_path = cfg.d("features", "features_meta.json")
    fingerprint = cfg.feature_fingerprint()
    if meta_path.is_file():
        try:
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("fingerprint") != fingerprint:
            # Settings changed -> descriptors are stale.  Remove them instead
            # of silently mixing SIFT runs with different parameters.
            for stale in cfg.d("features").glob("*.npz"):
                try:
                    stale.unlink()
                except OSError:
                    pass

    sift = cv2.SIFT_create(
        nfeatures=int(cfg.max_features),
        nOctaveLayers=int(cfg.sift_octave_layers),
        contrastThreshold=cfg.sift_contrast_threshold,
        edgeThreshold=10,
    )
    done = 0
    total_keypoints = 0
    reused = 0
    pressure: List[Dict[str, Any]] = []
    started = time.time()

    for index, frame in enumerate(frames, start=1):
        ctx.check()
        cached = load_features(cfg, frame)
        if cached is not None:
            reused += 1
            done += 1
            total_keypoints += int(cached["kp"].shape[0])
            continue

        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None:
            raise ReconstructionError(
                f"Feature extraction could not read {frame.name}.",
                suggestion="The frame is missing or corrupt; re-run frame "
                           "extraction to regenerate frames/original.",
                details={"file": str(frame)},
            )
        orig_h, orig_w = image.shape[:2]
        longest = max(orig_w, orig_h)
        if longest > cfg.max_image_size:
            scale = cfg.max_image_size / float(longest)
            image = cv2.resize(
                image,
                (max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        del gray, image

        if not keypoints or descriptors is None:
            # A textureless frame is possible (blank wall / sky).  Record it
            # instead of failing: matching will simply skip this image.
            kp = np.zeros((0, 5), dtype=np.float32)
            desc = np.zeros((0, 128), dtype=np.uint8)
        else:
            kp = np.array(
                [[k.pt[0], k.pt[1], k.size, k.angle, k.response] for k in keypoints],
                dtype=np.float32,
            )
            desc = descriptors.astype(np.uint8)

        np.savez_compressed(
            _npz_path(cfg, frame),
            kp=kp, desc=desc,
            width=np.int32(width), height=np.int32(height),
            orig_width=np.int32(orig_w), orig_height=np.int32(orig_h),
        )
        done += 1
        total_keypoints += int(kp.shape[0])
        del keypoints, descriptors, kp, desc

        if index % 5 == 0:
            pressure.append({"at_frame": index, **memory_pressure(cfg.ram_limit_gb)})
            ctx.note(f"{index}/{len(frames)} frames described "
                     f"({total_keypoints} keypoints, rss "
                     f"{pressure[-1]['process_rss_gb']} GB)")

    meta = {
        "fingerprint": fingerprint,
        "frames": done,
        "reused_from_disk": reused,
        "total_keypoints": total_keypoints,
        "average_per_image": round(total_keypoints / done, 1) if done else 0,
        "detector": "SIFT",
        "parameters": {
            "nfeatures": cfg.max_features,
            "nOctaveLayers": cfg.sift_octave_layers,
            "contrastThreshold": cfg.sift_contrast_threshold,
            "max_image_size": cfg.max_image_size,
            "threads": cfg.threads,
            "cpu_only": True,
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "memory_samples": pressure,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ctx.result(meta)
    if done == 0:
        raise ReconstructionError(
            "Feature extraction produced no descriptors.",
            suggestion="Frames may be textureless; supply a reference video "
                       "with visible vehicle detail.",
        )
    return meta
