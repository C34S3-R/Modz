"""PHASE 2b - sequential feature matching + RANSAC geometric verification.

Because the input is a video, frame N is matched against N+1 .. N+window
plus a longer keyframe hop that helps close an orbit (section 8).  Every
pair goes through Lowe's ratio test and then a RANSAC geometric model
(section 9): descriptors being numerically similar is never enough.

Outputs ``matches/<a>__<b>.npz`` per pair, so the stage resumes for free.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ReconConfig, ReconstructionError
from .features import load_features, selected_frames
from .state import StateStore, StageContext, memory_pressure

MODEL_NONE = 0
MODEL_ESSENTIAL = 1
MODEL_FUNDAMENTAL = 2


def _cv2():
    import cv2
    return cv2


def assumed_intrinsics(cfg: ReconConfig, width: int, height: int):
    """Pinhole intrinsics when the video carries no calibration.

    A horizontal field of view (default 60 deg, typical for phone/compact
    video) converts directly to a focal length in pixels.  EXIF is rarely
    present on video frames, so guessing the *angle* is far safer than
    guessing a pixel count: portrait and landscape frames then both start
    from something plausible.

    This is a *guess* and it cannot be fixed later: joint bundle adjustment
    has the points absorb any focal error (a flat direction), so the value
    stays where it starts.  ``sfm.run_sparse`` therefore tests several
    candidates with the focal multi-start before the real reconstruction,
    and ``focal_override_px`` is how it applies the winner here.
    """
    import math
    override = float(getattr(cfg, "focal_override_px", 0.0) or 0.0)
    if override > 0.0:          # focal multi-start already chose a value
        f = override
    else:
        fov = float(getattr(cfg, "initial_fov_degrees", 60.0) or 0.0)
        if 1.0 < fov < 179.0:
            f = 0.5 * width / math.tan(math.radians(fov) / 2.0)
        else:
            f = cfg.initial_focal_scale * float(max(width, height))
    import numpy as np
    K = np.array(
        [[f, 0.0, width / 2.0],
         [0.0, f, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K


def pair_key(a: str, b: str) -> str:
    return f"{a}__{b}"


def match_path(cfg: ReconConfig, a: str, b: str) -> Path:
    return cfg.d("matches", pair_key(a, b) + ".npz")


def list_match_files(cfg: ReconConfig) -> List[Path]:
    return sorted(cfg.d("matches").glob("*.npz"))


def load_match(cfg: ReconConfig, path: Path) -> Optional[Dict[str, Any]]:
    try:
        import numpy as np
        data = np.load(path, allow_pickle=False)
        return {
            "idx_a": data["idx_a"],
            "idx_b": data["idx_b"],
            "inlier": data["inlier"],
            "model": int(data["model"]),
            "matrix": data["matrix"],
            "a": str(data["a"]),
            "b": str(data["b"]),
        }
    except Exception:
        return None


def _pairs_for(frames: List[Path], cfg: ReconConfig) -> List[Tuple[int, int]]:
    """Sequential window pairs plus a few keyframe hops.

    Frame N is matched against N+1..N+window (section 8), and additionally
    against a sparse set of hops (N+4, N+6, ...).  The hops matter: feature
    tracks only become 3D points when two *registered* cameras share them,
    so links that skip a few frames are what let the reconstruction chain
    past a weak section of the video instead of stalling.
    """
    n = len(frames)
    pairs: set[Tuple[int, int]] = set()
    hops = sorted({4, 6, 9, 12, cfg.keyframe_skip, cfg.keyframe_skip * 2})
    for i in range(n):
        for offset in range(1, cfg.match_window + 1):
            j = i + offset
            if j < n:
                pairs.add((i, j))
        for hop in hops:
            j = i + hop
            if j < n:
                pairs.add((i, j))
    return sorted(pairs)


def extract_matches(cfg: ReconConfig, store: StateStore, ctx: StageContext) -> Dict[str, Any]:
    cv2 = _cv2()
    import numpy as np

    frames = selected_frames(cfg)
    if len(frames) < 2:
        raise ReconstructionError(
            f"Matching needs at least 2 frames; found {len(frames)}.",
            suggestion="Lower the frame interval or relax frame-quality "
                       "thresholds so more frames survive selection.",
        )
    cfg.d("matches").mkdir(parents=True, exist_ok=True)
    meta_path = cfg.d("matches", "matches_meta.json")
    fingerprint = cfg.match_fingerprint()
    # Deterministic verification: OpenCV's RANSAC draws from a global stream,
    # so two runs of the same input must reseed from the same place or the
    # inlier sets - and every track built from them - differ run to run.
    import zlib
    cv2.setRNGSeed(int(zlib.crc32(fingerprint.encode()) & 0x7FFFFFFF) or 1)
    if meta_path.is_file():
        try:
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("fingerprint") != fingerprint:
            for stale in list_match_files(cfg):
                try:
                    stale.unlink()
                except OSError:
                    pass

    names = [f.stem for f in frames]
    pairs = _pairs_for(frames, cfg)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # Rolling descriptor cache: only window+2 images are ever in memory.
    cache: "deque[Tuple[int, Optional[Dict[str, Any]]]]" = deque(maxlen=cfg.match_window + 3)

    def features_for(index: int) -> Optional[Dict[str, Any]]:
        for slot, (idx, feat) in enumerate(cache):
            if idx == index:
                return feat
        feat = load_features(cfg, frames[index])
        cache.append((index, feat))
        return feat

    stats = {
        "pairs_total": len(pairs),
        "pairs_verified": 0,
        "pairs_skipped_too_few": 0,
        "pairs_skipped_unverified": 0,
        "pairs_resumed_from_disk": 0,
        "total_raw_matches": 0,
        "total_inliers": 0,
    }
    per_pair: List[Dict[str, Any]] = []
    started = time.time()
    pressure: List[Dict[str, Any]] = []

    for count, (i, j) in enumerate(pairs, start=1):
        ctx.check()
        path = match_path(cfg, names[i], names[j])
        if path.is_file():
            cached = load_match(cfg, path)
            if cached is not None:
                n_inliers = int(cached["inlier"].sum())
                stats["pairs_resumed_from_disk"] += 1
                stats["total_inliers"] += n_inliers
                # Keep the per-pair record so a resumed run still reports the
                # whole match graph instead of only what it recomputed.
                per_pair.append({
                    "a": names[i], "b": names[j],
                    "raw": n_inliers, "good": n_inliers,
                    "inliers": n_inliers,
                    "model": "essential" if cached["model"] == MODEL_ESSENTIAL
                             else "fundamental",
                    "resumed": True,
                })
                continue

        fa, fb = features_for(i), features_for(j)
        if fa is None or fb is None or fa["desc"].shape[0] < 2 or fb["desc"].shape[0] < 2:
            stats["pairs_skipped_too_few"] += 1
            continue

        knn = matcher.knnMatch(fa["desc"], fb["desc"], k=2)
        good: List[Any] = []
        good_loose: List[Any] = []
        loose_ratio = cfg.ratio_test + 0.10
        for pair in knn:
            if len(pair) == 1:
                continue  # only one neighbour: cannot apply the ratio test
            m, n = pair
            if m.distance < cfg.ratio_test * n.distance:
                good.append(m)
            elif m.distance < loose_ratio * n.distance:
                good_loose.append(m)
        # Widen the ratio test only when the strict one found too little.
        # Frame pairs that overlap only partially (or repeat their own
        # texture) lose the tightest correspondences first; a looser cut is
        # still safe because RANSAC verifies the geometry immediately after.
        ratio_used = cfg.ratio_test
        if len(good) < cfg.min_matches and len(good_loose) >= cfg.min_matches:
            good = good + good_loose
            ratio_used = round(loose_ratio, 3)
        if len(good) < cfg.min_matches:
            stats["pairs_skipped_too_few"] += 1
            per_pair.append({"a": names[i], "b": names[j],
                             "raw": len(knn), "good": len(good),
                             "inliers": 0, "model": "none"})
            continue

        pts_a = np.float64([fa["kp"][m.queryIdx, :2] for m in good])
        pts_b = np.float64([fb["kp"][m.trainIdx, :2] for m in good])

        # --- geometric verification (section 9) ---------------------------
        # Verify with the fundamental matrix, estimated directly in pixel
        # coordinates.  Fitting an essential matrix through the assumed field
        # of view is *not* harmless: the principal-point term breaks the
        # scale invariance of the epipolar constraint (normalised points
        # transform as diag(k, k, 1), which no essential matrix survives), so
        # RANSAC keeps a K-dependent subset of correspondences and every
        # track inherits that bias.  Measured on the synthetic scene: the
        # same sequence landed 14% or 49% off ground truth depending only on
        # which FOV the matcher assumed.  Poses are re-derived from these
        # points in sfm.py using the focal length the reconstruction actually
        # settles on, so nothing downstream needs K to be right here.
        model = MODEL_NONE
        matrix = np.zeros((3, 3), dtype=np.float64)
        inlier_mask: Optional[np.ndarray] = None

        try:
            F, mask = cv2.findFundamentalMat(
                pts_a, pts_b, cv2.FM_RANSAC,
                cfg.ransac_threshold, 0.999,
            )
        except cv2.error:
            F, mask = None, None
        if F is not None and mask is not None and F.shape == (3, 3) \
                and int(mask.ravel().sum()) >= cfg.min_inliers:
            model = MODEL_FUNDAMENTAL
            matrix = F.astype(np.float64)
            inlier_mask = mask.ravel().astype(bool)
        else:
            # F degenerates on near-pure-rotation pairs; the rigid model can
            # still be fitted there, so keep it as a fallback.
            try:
                K = assumed_intrinsics(cfg, int(fa["width"]), int(fa["height"]))
                E, mask_e = cv2.findEssentialMat(
                    pts_a, pts_b, K, method=cv2.RANSAC,
                    prob=0.999, threshold=cfg.ransac_threshold,
                )
            except cv2.error:
                E, mask_e = None, None
            if E is not None and mask_e is not None and E.shape == (3, 3) \
                    and int(mask_e.ravel().sum()) >= cfg.min_inliers:
                model = MODEL_ESSENTIAL
                matrix = E.astype(np.float64)
                inlier_mask = mask_e.ravel().astype(bool)

        if inlier_mask is None:
            stats["pairs_skipped_unverified"] += 1
            per_pair.append({"a": names[i], "b": names[j],
                             "raw": len(knn), "good": len(good),
                             "inliers": 0, "model": "rejected"})
            continue

        idx_a = np.array([m.queryIdx for m in good], dtype=np.int32)[inlier_mask]
        idx_b = np.array([m.trainIdx for m in good], dtype=np.int32)[inlier_mask]
        np.savez_compressed(
            path,
            idx_a=idx_a, idx_b=idx_b,
            inlier=np.ones(len(idx_a), dtype=bool),
            model=np.int32(model), matrix=matrix,
            a=np.str_(names[i]), b=np.str_(names[j]),
        )
        stats["pairs_verified"] += 1
        stats["total_raw_matches"] += len(good)
        stats["total_inliers"] += int(len(idx_a))
        per_pair.append({"a": names[i], "b": names[j],
                         "raw": len(knn), "good": len(good),
                         "inliers": int(len(idx_a)),
                         "ratio": ratio_used,
                         "model": "essential" if model == MODEL_ESSENTIAL else "fundamental"})

        if count % 10 == 0:
            pressure.append({"at_pair": count, **memory_pressure(cfg.ram_limit_gb)})
            ctx.note(f"{count}/{len(pairs)} pairs verified "
                     f"({stats['total_inliers']} inliers, rss "
                     f"{pressure[-1]['process_rss_gb']} GB)")

    # A fully cached re-run verifies nothing *new*, which must not be read as
    # a failure: resumed pairs already carry their inlier masks.
    if stats["pairs_verified"] == 0 and stats["pairs_resumed_from_disk"] == 0:
        raise ReconstructionError(
            f"Geometric verification rejected all {len(pairs)} image pairs.",
            suggestion="The frames may be from a static scene with too little "
                       "camera movement, or too blurry. Use a video that orbits "
                       "the vehicle, or lower --frame-interval to 0.25 so "
                       "neighbouring frames overlap more.",
            details={"pairs_total": len(pairs),
                     "skipped_too_few": stats["pairs_skipped_too_few"],
                     "skipped_unverified": stats["pairs_skipped_unverified"]},
        )

    meta = {
        "fingerprint": fingerprint,
        **stats,
        "parameters": {
            "match_window": cfg.match_window,
            "ratio_test": cfg.ratio_test,
            "min_matches": cfg.min_matches,
            "min_inliers": cfg.min_inliers,
            "ransac_threshold_px": cfg.ransac_threshold,
            "keyframe_skip": cfg.keyframe_skip,
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "pairs": per_pair,
        "memory_samples": pressure,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ctx.result(stats)
    return meta
