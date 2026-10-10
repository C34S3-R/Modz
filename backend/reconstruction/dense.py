"""PHASE 4 - dense depth maps by plane-sweep multi-view block matching (CPU).

For every registered view the sweep warps each neighbour view through a
family of per-pixel depth hypotheses, scores agreement with masked local NCC,
and keeps the hypothesis that best explains the pixel.  Reference views are
swept in parallel worker threads (ThreadPoolExecutor, bounded by
``dense_workers``) and each depth map is written to disk by the main thread
as results land, so the E6410 holds at most ``workers`` working-size image
sets plus one (planes x pixels) cost volume each - measured peak stays
around 0.5 GB per worker.

Three design rules carry the accuracy; each was measured against the sparse
reconstruction used as ground truth (projects/realvid, median relative depth
error at sparse points):

1. *Per-pixel inverse-depth ranges.*  One global range has to cover the
   vehicle and the far background at once, so its samples are far too coarse
   where the vehicle is: 4% log spacing left a 6 px disparity error at
   baseline 3.7 - wider than the NCC window - and depth error reached 35%.
   Each 16 px block instead gets its own [z_lo, z_hi] from the sparse points
   landing in it; empty blocks inherit a neighbour's range.

2. *Planes uniform in inverse depth, count from the disparity budget.*
   Constant inverse-depth spacing means a constant disparity step, and the
   plane count is chosen so that step stays within ``dense_step_px`` (1.5 px)
   - what the NCC window can still match.  Measured: 2.3% median error where
   the coarse global sweep gave 35%.

3. *Neighbours inside a resolvable parallax window.*  A view helps only if
   its baseline moves the image by more than ``dense_min_parallax`` (too
   similar contributes no depth signal) and no more than the plane budget can
   resolve (too far breaks the match).  Both tests are ratios against the
   scene's own depth, so they hold in any reconstruction scale.

The stage reports an honest cross-check - dense depth sampled at the sparse
points' pixels versus those points' own depth - plus per-view coverage and
match confidence, so a partially textured video shows up as missing views
instead of silent holes.

Outputs
-------
``dense/<frame>.npz``           depth, ncc, agreement, valid mask, scale
``dense/dense_point_cloud.ply`` fused, voxel-downsampled coloured cloud
``dense/dense_meta.json``       parameters, per-view stats, validation
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ReconConfig, ReconstructionError
from .ply import read_ply, write_ply
from .state import StateStore, StageContext, memory_pressure

# Exported cloud density: voxels across the scene diagonal.  Deliberately a
# constant, not a knob - the cloud is the handover artefact for inspection,
# and the mesh stage re-samples the depth maps rather than trusting it.
_CLOUD_VOXEL_DIVISOR = 384.0
_MAX_FUSION_POINTS = 4_000_000     # hard memory guard before voxelising
_MIN_VIEW_POINTS = 50              # sparse points needed to sweep one view


def _cv2():
    import cv2
    return cv2


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_views(cfg: ReconConfig) -> List[Dict[str, Any]]:
    """Registered camera poses from the sparse reconstruction."""
    sfm_path = cfg.d("sfm", "sfm.json")
    if not sfm_path.is_file():
        raise ReconstructionError(
            "Dense reconstruction needs the sparse reconstruction "
            "(sfm/sfm.json).",
            suggestion="Run the pipeline through the 'sfm' stage first; the "
                       "dense stage reads poses and the sparse cloud from it.",
            details={"file": str(sfm_path)},
        )
    payload = json.loads(sfm_path.read_text(encoding="utf-8"))
    views: List[Dict[str, Any]] = []
    for entry in payload.get("images", []):
        if not entry.get("registered", True):
            continue
        R = np.array(entry["R"], dtype=np.float64).reshape(3, 3)
        t = np.array(entry["t"], dtype=np.float64)
        width, height = int(entry["width"]), int(entry["height"])
        focal = float(entry["focal"])
        views.append(
            {
                "name": entry["name"],
                "R": R,
                "t": t,
                "width": width,
                "height": height,
                "focal": focal,
                "K": np.array(
                    [[focal, 0.0, width / 2.0],
                     [0.0, focal, height / 2.0],
                     [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
                "center": -R.T @ t,
            }
        )
    if len(views) < 2:
        raise ReconstructionError(
            f"Dense reconstruction needs at least 2 registered views, found "
            f"{len(views)}.",
            suggestion="The sparse stage did not register enough images. "
                       "Check reconstruction_report.json for "
                       "unregistered_reasons.",
        )
    return views


def _sparse_cloud(cfg: ReconConfig) -> np.ndarray:
    path = cfg.d("sparse", "sparse_point_cloud.ply")
    if not path.is_file():
        raise ReconstructionError(
            "Dense reconstruction needs the sparse point cloud "
            "(sparse/sparse_point_cloud.ply).",
            suggestion="Run the 'sfm' stage first; it exports the cloud the "
                       "dense stage uses for depth ranges.",
            details={"file": str(path)},
        )
    xyz, _rgb = read_ply(path)
    if len(xyz) < 50:
        raise ReconstructionError(
            f"Sparse cloud has only {len(xyz)} points; too few to derive "
            f"depth ranges for the dense sweep.",
            suggestion="Re-run the sparse stage or supply a video with more "
                       "overlap between frames.",
        )
    return xyz.astype(np.float64)


def _frames_by_name(cfg: ReconConfig) -> Dict[str, Path]:
    from .features import selected_frames

    masked = sorted(cfg.d("frames", "masked").glob("frame_*.jpg"))
    if masked:
        # Geometry stages (dense, mesh, texture) consume the masked copies
        # so the background contributes no depth or texels.  Pose stages
        # read frames/selected via selected_frames() directly.
        return {path.stem: path for path in masked}
    return {path.stem: path for path in selected_frames(cfg)}


def _depth_path(cfg: ReconConfig, name: str) -> Path:
    return cfg.d("dense", name + ".npz")


def _load_depth(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load a depth map, tolerating a half-written file from a killed run."""
    try:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError, EOFError):
        try:
            path.unlink()
        except OSError:
            pass
        return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _project(view: Dict[str, Any], points: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> (u, v, z) in ``view``; z <= 0 is behind the camera."""
    cam = view["R"] @ points.T + view["t"][:, None]
    z = cam[2]
    denom = np.where(np.abs(z) > 1e-9, z, 1e-9)
    uv = view["K"] @ cam
    return uv[0] / denom, uv[1] / denom, z


def _grid_depth_range(pts_ref: np.ndarray, height: int, width: int,
                      block: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Per-pixel [z_lo, z_hi] built from sparse points on a coarse grid.

    Blocks without points inherit from neighbours (min for the near bound,
    max for the far bound) so every pixel gets a usable range; only a view
    with no points at all returns None.
    """
    if len(pts_ref) == 0:
        return None
    gh, gw = (height + block - 1) // block, (width + block - 1) // block
    zlo = np.full((gh, gw), np.inf, np.float64)
    zhi = np.full((gh, gw), -np.inf, np.float64)
    bx = np.clip((pts_ref[:, 0] // block).astype(np.int64), 0, gw - 1)
    by = np.clip((pts_ref[:, 1] // block).astype(np.int64), 0, gh - 1)
    z = pts_ref[:, 2]
    np.minimum.at(zlo, (by, bx), z)
    np.maximum.at(zhi, (by, bx), z)

    filled = np.isfinite(zlo)
    for _ in range(max(gh, gw)):
        if filled.all():
            break
        lo_src = np.where(filled, zlo, np.inf)
        hi_src = np.where(filled, zhi, -np.inf)
        cand_lo = lo_src.copy()
        cand_hi = hi_src.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                cand_lo = np.minimum(cand_lo, np.roll(np.roll(lo_src, dy, 0), dx, 1))
                cand_hi = np.maximum(cand_hi, np.roll(np.roll(hi_src, dy, 0), dx, 1))
        newly = (~filled) & (cand_lo < np.inf)
        if not newly.any():
            break
        zlo = np.where(newly, cand_lo, zlo)
        zhi = np.where(newly, cand_hi, zhi)
        filled |= newly
    if not filled.all():
        zlo = np.where(np.isfinite(zlo), zlo, float(z.min()))
        zhi = np.where(np.isfinite(zhi), zhi, float(z.max()))

    cv2 = _cv2()
    zlo_full = cv2.resize(zlo.astype(np.float32), (width, height),
                          interpolation=cv2.INTER_NEAREST)
    zhi_full = cv2.resize(zhi.astype(np.float32), (width, height),
                          interpolation=cv2.INTER_NEAREST)
    # A block can contain both near and far points; keep the bounds ordered
    # and positive so the sweep always has a sane interval.
    zlo_full = np.clip(zlo_full, 1e-3, None)
    zhi_full = np.maximum(zhi_full, zlo_full * 1.0001)
    return zlo_full, zhi_full


def _select_neighbors(views: List[Dict[str, Any]], ref_idx: int, *,
                      median_depth: float, inv_span_max: float,
                      focal_work: float,
                      cfg: ReconConfig) -> List[Tuple[int, float]]:
    """Neighbours whose parallax is non-degenerate *and* resolvable.

    ``span = f * B * (1/z_lo - 1/z_hi)`` is how far one hypothesis can move
    the sampled pixel - the disparity the plane budget must cover.  Both
    gates are ratios against the scene's own depth, so the same defaults
    work for any reconstruction scale.  Returns [(view_index, span)].
    """
    ref = views[ref_idx]
    candidates: List[Tuple[float, int, float]] = []
    for index, view in enumerate(views):
        if index == ref_idx:
            continue
        baseline = float(np.linalg.norm(view["center"] - ref["center"]))
        if baseline <= 1e-9:
            continue
        parallax = baseline / max(median_depth, 1e-9)
        if parallax < cfg.dense_min_parallax:
            continue
        span = focal_work * baseline * inv_span_max
        if span > cfg.dense_max_planes * cfg.dense_step_px:
            continue
        candidates.append((parallax, index, span))
    # Closest first: least occlusion and appearance change, which is what
    # NCC wants; the parallax floor keeps near-duplicates from dominating.
    candidates.sort()
    chosen = [(index, span) for _, index, span in
              candidates[: max(1, cfg.dense_neighbors)]]
    return chosen


def _reference_ncc_terms(gray_ref: np.ndarray, window: int):
    """Reference-side box-filter terms for NCC; constant for a whole view.

    Only the *warped* side changes per hypothesis, so the reference means and
    second moments are computed once here instead of six times per plane.
    """
    cv2 = _cv2()
    kernel = (int(window), int(window))
    a = gray_ref.astype(np.float32, copy=False)
    return a, cv2.blur(a, kernel), cv2.blur(a * a, kernel), kernel


def _plane_ncc(a: np.ndarray, ma: np.ndarray, maa: np.ndarray,
               warped: np.ndarray, coverage: np.ndarray, kernel) -> np.ndarray:
    """NCC between the reference and a warped source hypothesis.

    Unmasked box filters are exact wherever the whole support window is
    valid, so windows that touch invalid pixels (``coverage < 1``) are
    rejected outright rather than mixed with zero padding.
    """
    cv2 = _cv2()
    mab = cv2.blur(a * warped, kernel)
    mb = cv2.blur(warped, kernel)
    mbb = cv2.blur(warped * warped, kernel)
    var_a = np.maximum(maa - ma * ma, np.float32(1e-4))
    var_b = np.maximum(mbb - mb * mb, np.float32(1e-4))
    ncc = (mab - ma * mb) / np.sqrt(var_a * var_b)
    ncc = np.clip(ncc, -1.0, 1.0)
    return np.where(coverage >= np.float32(1.0 - 1e-6), ncc,
                    np.float32(-1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _working_size(width: int, height: int, cfg: ReconConfig) -> Tuple[int, int]:
    scale = min(cfg.dense_max_size / float(max(width, height)), 1.0)
    return (max(1, int(round(width * scale))),
            max(1, int(round(height * scale))))


def _load_gray(path: Path, target: Tuple[int, int]) -> np.ndarray:
    cv2 = _cv2()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ReconstructionError(
            f"Dense reconstruction could not read frame {path.name}.",
            suggestion="The selected frame is missing or corrupt; re-run "
                       "frame extraction.",
            details={"file": str(path)},
        )
    if (image.shape[1], image.shape[0]) != target:
        image = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)


def _load_color(path: Path, target: Tuple[int, int]) -> np.ndarray:
    cv2 = _cv2()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ReconstructionError(
            f"Dense reconstruction could not read frame {path.name}.",
            suggestion="The selected frame is missing or corrupt; re-run "
                       "frame extraction.",
            details={"file": str(path)},
        )
    if (image.shape[1], image.shape[0]) != target:
        image = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    return image


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def _working_intrinsics(view: Dict[str, Any], target: Tuple[int, int]
                        ) -> np.ndarray:
    """K for the working-resolution image (principal point at its centre)."""
    width, height = target
    focal = float(view["focal"]) * (width / float(view["width"]))
    return np.array(
        [[focal, 0.0, width / 2.0],
         [0.0, focal, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _sweep_view(cfg: ReconConfig, ref: Dict[str, Any],
                sources: List[Tuple[Dict[str, Path], Path]],
                gray_ref: np.ndarray, rays: np.ndarray,
                inv_lo: np.ndarray, inv_hi: np.ndarray,
                planes: int, ctx: StageContext
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep one reference view against its neighbours.

    ``sources`` is [(view, frame_path)]; ``rays`` the (3, H*W) reference ray
    directions (z = 1).  Returns (depth, ncc, agreement) at working size:
    ``ncc`` is the best match confidence seen for the pixel, ``agreement``
    the share of *confident* neighbours whose own argmin sits next to the
    fused one.

    The per-plane math runs in float32 with an explicit projection instead
    of BLAS matmuls: at working resolution that is exact to well under a
    thousandth of a pixel and about twice as fast on this CPU.
    """
    cv2 = _cv2()
    height, width = gray_ref.shape
    n_pixels = height * width
    cost = np.zeros((planes, n_pixels), dtype=np.float32)
    source_maps: List[Tuple[np.ndarray, np.ndarray]] = []   # (argmin, best ncc)
    best_ncc = np.full(n_pixels, -1.0, np.float32)

    ref_a, ref_ma, ref_maa, kernel = _reference_ncc_terms(
        gray_ref, cfg.dense_ncc_window)

    inv_lo_r = inv_lo.astype(np.float32, copy=False).reshape(-1)
    inv_hi_r = inv_hi.astype(np.float32, copy=False).reshape(-1)
    inv_span = inv_hi_r - inv_lo_r
    rays32 = np.ascontiguousarray(rays, dtype=np.float32)
    plane_t = np.arange(planes, dtype=np.float32) / max(planes - 1, 1)
    one = np.float32(1.0)

    for source_view, frame_path in sources:
        ctx.check()
        target = _working_size(source_view["width"], source_view["height"], cfg)
        gray_src = _load_gray(frame_path, target)
        Ks = _working_intrinsics(source_view, target)
        R64 = source_view["R"] @ ref["R"].T
        t64 = source_view["t"] - R64 @ ref["t"]
        R = R64.astype(np.float32)
        t = t64.astype(np.float32)
        f = np.float32(Ks[0, 0])
        cx = np.float32(Ks[0, 2])
        cy = np.float32(Ks[1, 2])

        src_best_ncc = np.full(n_pixels, -1.0, np.float32)
        src_best_am = np.zeros(n_pixels, np.int32)
        src_h, src_w = gray_src.shape
        max_x = np.float32(src_w - 1)
        max_y = np.float32(src_h - 1)

        for plane_index in range(planes):
            if plane_index % 16 == 0:
                ctx.check()
            invd = inv_lo_r + plane_t[plane_index] * inv_span   # per pixel
            depth_h = one / np.maximum(invd, np.float32(1e-9))
            c0 = rays32[0] * depth_h
            c1 = rays32[1] * depth_h
            c2 = rays32[2] * depth_h
            # ref camera -> source camera, then pinhole projection
            x = (R[0, 0] * c0 + R[0, 1] * c1 + R[0, 2] * c2) + t[0]
            y = (R[1, 0] * c0 + R[1, 1] * c1 + R[1, 2] * c2) + t[1]
            z = (R[2, 0] * c0 + R[2, 1] * c1 + R[2, 2] * c2) + t[2]
            ok = z > np.float32(1e-6)
            if not ok.any():
                cost[plane_index] += 2.0        # nowhere observable
                continue
            zs = np.where(ok, z, one)
            mapx = (f * x) / zs + cx
            mapy = (f * y) / zs + cy
            valid = (ok & (mapx >= 0) & (mapx <= max_x)
                     & (mapy >= 0) & (mapy <= max_y))
            if not valid.any():
                cost[plane_index] += 2.0        # fully out of the source view
                continue
            mapx = mapx.reshape(height, width)
            mapy = mapy.reshape(height, width)
            valid2 = valid.reshape(height, width)
            warped = cv2.remap(gray_src, mapx, mapy, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            coverage = cv2.blur(valid2.astype(np.float32), kernel)
            ncc = _plane_ncc(ref_a, ref_ma, ref_maa, warped, coverage, kernel)
            ncc = np.where(valid2, ncc, np.float32(-1.0))
            # Invalid hypotheses cost the maximum (2.0): a plane the warp
            # cannot even sample must never win the argmin.
            cost[plane_index] += np.maximum(one - ncc, np.float32(0)).reshape(-1)
            flat = ncc.reshape(-1)
            upd = flat > src_best_ncc
            src_best_ncc = np.where(upd, flat, src_best_ncc)
            src_best_am = np.where(upd, np.int32(plane_index), src_best_am)
        best_ncc = np.maximum(best_ncc, src_best_ncc)
        source_maps.append((src_best_am, src_best_ncc))

    fused_am = np.argmin(cost, axis=0)

    # Sub-pixel refinement: parabola through the three costs around the
    # argmin, interpolated in the inverse-depth parameter (uniform spacing).
    pixels = np.arange(n_pixels)
    lo_i = np.clip(fused_am - 1, 0, planes - 1)
    hi_i = np.clip(fused_am + 1, 0, planes - 1)
    y0 = cost[lo_i, pixels]
    y1 = cost[fused_am, pixels]
    y2 = cost[hi_i, pixels]
    denom = y0 - 2.0 * y1 + y2
    shift = np.where(np.abs(denom) > np.float32(1e-6),
                     0.5 * (y0 - y2) / np.where(np.abs(denom) > np.float32(1e-6),
                                                denom, one),
                     np.float32(0.0))
    shift = np.clip(shift, -0.5, 0.5)
    refined = np.clip(fused_am + shift, 0, planes - 1)

    t_val = refined / np.float32(max(planes - 1, 1))
    invd = inv_lo_r + t_val * inv_span
    depth = (one / np.maximum(invd, np.float32(1e-9))).reshape(height, width)

    # Neighbour agreement: among neighbours that found a confident match for
    # this pixel, the share whose own argmin sits within two planes of the
    # fused answer (a tolerance tied to the ~1.5 px disparity step).
    tolerance = 2
    agree_num = np.zeros(n_pixels, np.float32)
    agree_den = np.zeros(n_pixels, np.float32)
    for src_am, src_ncc in source_maps:
        participates = src_ncc >= cfg.dense_min_ncc
        agree_den += participates.astype(np.float32)
        agree_num += (participates
                      & (np.abs(src_am - fused_am) <= tolerance)).astype(np.float32)
    agreement = np.where(agree_den > 0,
                         agree_num / np.maximum(agree_den, one),
                         np.float32(0.0))

    return (depth.astype(np.float32),
            best_ncc.reshape(height, width),
            agreement.reshape(height, width))


def _sparse_agreement(depth: np.ndarray, pts_ref: np.ndarray) -> Dict[str, float]:
    """Dense depth sampled at sparse-point pixels vs the sparse depth itself.

    This is the stage's validation metric: the sparse reconstruction is the
    independent reference, so disagreement is reported, not hidden.
    """
    if len(pts_ref) == 0:
        return {"samples": 0}
    height, width = depth.shape
    xi = np.round(pts_ref[:, 0]).astype(np.int64)
    yi = np.round(pts_ref[:, 1]).astype(np.int64)
    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    if not inside.any():
        return {"samples": 0}
    xi, yi, truth = xi[inside], yi[inside], pts_ref[inside, 2]
    est = depth[yi, xi].astype(np.float64)
    usable = est > 0
    if not usable.any():
        return {"samples": 0}
    rel = np.abs(est[usable] - truth[usable]) / np.maximum(truth[usable], 1e-9)
    return {
        "samples": int(usable.sum()),
        "median_rel_err": round(float(np.median(rel)), 4),
        "mean_rel_err": round(float(rel.mean()), 4),
    }


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def _fuse_cloud(cfg: ReconConfig, views: List[Dict[str, Any]],
                frames: Dict[str, Path], ctx: StageContext
                ) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Backproject every confident depth map and voxel-downsample."""
    xyz_chunks: List[np.ndarray] = []
    rgb_chunks: List[np.ndarray] = []
    total = 0
    for view in views:
        data = _load_depth(_depth_path(cfg, view["name"]))
        if data is None:
            continue
        depth = data["depth"]
        valid = data["valid"].astype(bool)
        if not valid.any():
            continue
        height, width = depth.shape
        target = (width, height)
        idx = np.flatnonzero(valid.reshape(-1))
        K_w = _working_intrinsics(view, target)
        ys, xs = np.mgrid[0:height, 0:width]
        grid = np.stack([xs.reshape(-1), ys.reshape(-1),
                         np.ones(height * width, np.float64)])
        rays = np.linalg.inv(K_w) @ grid
        d = depth.reshape(-1)[idx].astype(np.float64)
        cam = rays[:, idx] * d[None, :]
        world = view["R"].T @ (cam - view["t"][:, None])
        color = _load_color(frames[view["name"]], target)
        flat = color.reshape(-1, 3)[idx]
        xyz_chunks.append(world.T.astype(np.float32))
        rgb_chunks.append(flat[:, ::-1].astype(np.uint8))   # BGR -> RGB
        total += len(idx)
        ctx.check()

    stats: Dict[str, Any] = {
        "points_raw": total,
        "points_fused": 0,
        "voxel_size": None,
        "subsampled": False,
    }
    if not xyz_chunks:
        return None, stats

    pts = np.concatenate(xyz_chunks, axis=0)
    rgb = np.concatenate(rgb_chunks, axis=0)
    del xyz_chunks, rgb_chunks
    if len(pts) > _MAX_FUSION_POINTS:
        rng = np.random.default_rng(0)          # deterministic subsample
        keep = rng.choice(len(pts), _MAX_FUSION_POINTS, replace=False)
        pts, rgb = pts[keep], rgb[keep]
        stats["subsampled"] = True

    # Size the voxels from the *bulk* of the cloud: a few confident-looking
    # but far-away pixels (sky inside a propagated depth range) would
    # otherwise stretch the bounding box until the vehicle itself is only a
    # handful of voxels across.  p1/p99 bounds drop exactly those.
    lo = np.percentile(pts, 1.0, axis=0)
    hi = np.percentile(pts, 99.0, axis=0)
    pad = (hi - lo) * 0.05
    keep = np.all((pts >= lo - pad) & (pts <= hi + pad), axis=1)
    dropped = int((~keep).sum())
    if len(pts) - dropped >= 1000:
        pts, rgb = pts[keep], rgb[keep]
        stats["points_outlier_dropped"] = dropped
    else:
        stats["points_outlier_dropped"] = 0     # odd distribution; keep all

    extent = pts.max(axis=0) - pts.min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    voxel = max(diagonal / _CLOUD_VOXEL_DIVISOR, 1e-6)
    stats["voxel_size"] = round(voxel, 4)
    keys = np.floor((pts - pts.min(axis=0)) / voxel).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx.sort()
    fused_pts = pts[unique_idx]
    fused_rgb = rgb[unique_idx]
    stats["points_fused"] = int(len(fused_pts))

    out = write_ply(cfg.d("dense", "dense_point_cloud.ply"), fused_pts, fused_rgb)
    ctx.add_output_files([out])
    return out, stats


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

class _CheckShim:
    """Minimal ctx for worker threads: cancellation only, no logging."""

    def __init__(self, check: Any = None):
        self._check = check

    def check(self) -> None:
        if self._check is not None:
            self._check()


def _sweep_one_view(cfg: ReconConfig, views: List[Dict[str, Any]],
                    points: np.ndarray, frames: Dict[str, Path],
                    ref_index: int, check: Any = None) -> Dict[str, Any]:
    """Sweep a single reference view (worker-thread entry point).

    Pure compute: reads shared views/points/frames, returns either a
    ``skipped`` record or the arrays for the main thread to persist.
    Only ``check`` (cancellation) touches pipeline state from workers.
    """
    ref = views[ref_index]
    if check is not None:
        check()
    out_name = ref["name"]
    frame_path = frames.get(out_name)

    # --- reference geometry ------------------------------------------
    u, v, z = _project(ref, points)
    inside = ((z > 0.05) & np.isfinite(u) & np.isfinite(v)
              & (u >= 0) & (u < ref["width"])
              & (v >= 0) & (v < ref["height"]))
    pts_full = np.stack([u[inside], v[inside], z[inside]], axis=1)
    if frame_path is None or len(pts_full) < _MIN_VIEW_POINTS:
        return {"view": out_name, "outcome": "skipped",
                "reason": ("frame missing" if frame_path is None
                           else f"only {len(pts_full)} sparse points in frustum")}

    target = _working_size(ref["width"], ref["height"], cfg)
    gray_ref = _load_gray(frame_path, target)
    height, width = gray_ref.shape
    sx, sy = width / ref["width"], height / ref["height"]
    pts_ref = np.stack([pts_full[:, 0] * sx, pts_full[:, 1] * sy,
                        pts_full[:, 2]], axis=1)

    ranges = _grid_depth_range(pts_ref, height, width, cfg.dense_block)
    if ranges is None:
        return {"view": out_name, "outcome": "skipped",
                "reason": "no usable depth range"}
    zlo, zhi = ranges
    inv_lo = 1.0 / zhi
    inv_hi = 1.0 / zlo
    # Robust (p95) rather than max: one block containing a near-camera
    # sparse outlier (measured: a point at z=0.66 next to background at
    # z=75) would demand an unaffordable plane count for the whole view
    # and, through the span gate, shut out every neighbour.  The few
    # extreme blocks are simply swept a little coarser.
    inv_span_max = float(np.percentile(inv_hi - inv_lo, 95))
    median_depth = float(np.median(pts_full[:, 2]))

    K_w = _working_intrinsics(ref, target)
    focal_work = float(K_w[0, 0])
    chosen = _select_neighbors(
        views, ref_index, median_depth=median_depth,
        inv_span_max=inv_span_max, focal_work=focal_work, cfg=cfg)
    if not chosen:
        return {"view": out_name, "outcome": "skipped",
                "reason": "no neighbour within resolvable parallax"}

    span_max = max(span for _, span in chosen)
    planes = int(np.clip(np.ceil(span_max / cfg.dense_step_px),
                         32, cfg.dense_max_planes))

    ys, xs = np.mgrid[0:height, 0:width]
    grid = np.stack([xs.reshape(-1), ys.reshape(-1),
                     np.ones(height * width, np.float64)])
    rays = np.linalg.inv(K_w) @ grid

    sources = []
    for index, _span in chosen:
        src_path = frames.get(views[index]["name"])
        if src_path is None:
            continue
        sources.append((views[index], src_path))
    if len(sources) == 0:
        return {"view": out_name, "outcome": "skipped",
                "reason": "neighbour frames missing"}

    depth, ncc, agreement = _sweep_view(
        cfg, ref, sources, gray_ref, rays, inv_lo, inv_hi, planes,
        _CheckShim(check))

    agreement_check = _sparse_agreement(depth, pts_ref)
    valid = ncc >= cfg.dense_min_ncc
    return {
        "view": out_name, "outcome": "swept",
        "neighbors": len(sources), "planes": planes,
        "depth": depth.astype(np.float32), "ncc": ncc.astype(np.float32),
        "agreement": agreement.astype(np.float32),
        "valid": valid.astype(np.uint8), "sx": np.float32(sx),
        "width": np.int32(width), "height": np.int32(height),
        "coverage": round(float(valid.mean()), 4),
        "ncc_mean": round(float(ncc[valid].mean()), 4) if valid.any() else 0.0,
        "agreement_mean": round(float(agreement[valid].mean()), 4)
        if valid.any() else 0.0,
        "sparse_check": agreement_check,
    }


def run_dense(cfg: ReconConfig, store: StateStore, ctx: StageContext) -> Dict[str, Any]:
    cv2 = _cv2()  # noqa: F841  (import applies CPU/thread policy from config)
    views = load_views(cfg)
    points = _sparse_cloud(cfg)
    frames = _frames_by_name(cfg)
    cfg.d("dense").mkdir(parents=True, exist_ok=True)

    fingerprint = cfg.dense_fingerprint()
    meta_path = cfg.d("dense", "dense_meta.json")
    if meta_path.is_file():
        try:
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("fingerprint") != fingerprint:
            # Sweep settings changed -> depth maps are stale.  Remove them
            # instead of silently mixing runs with different hypotheses.
            removed = 0
            for stale in cfg.d("dense").glob("*.npz"):
                try:
                    stale.unlink()
                    removed += 1
                except OSError:
                    pass
            if removed:
                ctx.note(f"settings changed; dropped {removed} stale depth maps")

    ctx.set_command(
        f"in-process plane sweep (OpenCV CPU, {len(views)} views, "
        f"<= {cfg.dense_neighbors} neighbours each)")
    ctx.log(f"dense sweep over {len(views)} registered views "
            f"(working size {cfg.dense_max_size}px, step {cfg.dense_step_px}px)")

    started = time.time()
    pressure: List[Dict[str, Any]] = []
    per_view: List[Dict[str, Any]] = []
    reused = 0
    skipped: List[Dict[str, str]] = []
    agreement_rel: List[float] = []

    # --- parallel sweep --------------------------------------------------
    # Views are independent: each reads the shared views/points/frames and
    # produces its own depth map.  Cached maps are reused up front; the rest
    # run in worker threads while the main thread persists results and owns
    # all logging/progress.  A missing frame now skips its view instead of
    # failing the stage - the map count in the summary says what happened.
    import concurrent.futures as _futures

    for ref_index, ref in enumerate(views):
        ctx.check()
        out_path = _depth_path(cfg, ref["name"])
        if out_path.is_file():
            cached = _load_depth(out_path)
            if cached is not None:
                reused += 1
                valid = cached["valid"].astype(bool)
                per_view.append(
                    {
                        "view": ref["name"],
                        "reused": True,
                        "coverage": round(float(valid.mean()), 4),
                        "ncc_mean": round(float(cached["ncc"][valid].mean()), 4)
                        if valid.any() else 0.0,
                    }
                )
                continue

    todo = [i for i in range(len(views))
            if not _depth_path(cfg, views[i]["name"]).is_file()]
    workers = max(1, min(3, int(getattr(cfg, "threads", 2) or 2)))
    ctx.note(f"sweeping {len(todo)} views on {workers} workers "
             f"({reused} reused from cache)")
    done_count = 0
    check = ctx.check
    with _futures.ThreadPoolExecutor(max_workers=workers,
                                     thread_name_prefix="sweep") as pool:
        future_of = {pool.submit(_sweep_one_view, cfg, views, points,
                                 frames, i, check): i for i in todo}
        for future in _futures.as_completed(future_of):
            ctx.check()
            result = future.result()
            done_count += 1
            if result["outcome"] == "skipped":
                skipped.append({"view": result["view"],
                                "reason": result["reason"]})
                ctx.note(f"{result['view']}: skipped ({result['reason']})")
                continue
            out_path = _depth_path(cfg, result["view"])
            np.savez_compressed(
                out_path,
                depth=result["depth"],
                ncc=result["ncc"],
                agreement=result["agreement"],
                valid=result["valid"],
                scale=result["sx"],
                width=result["width"],
                height=result["height"],
            )
            if result["sparse_check"].get("samples"):
                agreement_rel.append(
                    result["sparse_check"]["median_rel_err"])
            per_view.append(
                {
                    "view": result["view"],
                    "reused": False,
                    "neighbors": result["neighbors"],
                    "planes": result["planes"],
                    "coverage": result["coverage"],
                    "ncc_mean": result["ncc_mean"],
                    "agreement_mean": result["agreement_mean"],
                    "sparse_check": result["sparse_check"],
                }
            )
            if done_count % 5 == 0 or done_count == len(todo):
                sample = {"at_view": done_count,
                          **memory_pressure(cfg.ram_limit_gb)}
                pressure.append(sample)
                ctx.note(
                    f"{done_count}/{len(todo)} views swept "
                    f"({reused} reused, rss {sample['process_rss_gb']} GB, "
                    f"last coverage {per_view[-1]['coverage']:.0%})")
    swept = [entry for entry in per_view if not entry.get("reused")]
    if not per_view:
        raise ReconstructionError(
            "Dense reconstruction produced no depth maps.",
            suggestion="No registered view had enough sparse points or a "
                       "resolvable neighbour. See reconstruction_report.json "
                       "for the sparse stage's coverage.",
            details={"skipped": skipped[:20]},
        )

    ctx.check()
    cloud_path, cloud_stats = _fuse_cloud(cfg, views, frames, ctx)

    coverages = [entry["coverage"] for entry in per_view]
    nccs = [entry["ncc_mean"] for entry in per_view if entry.get("ncc_mean")]
    meta: Dict[str, Any] = {
        "fingerprint": fingerprint,
        "algorithm": "plane sweep, masked local NCC, inverse-depth hypotheses",
        "views_total": len(views),
        "views_with_depth": len(per_view),
        "views_reused": reused,
        "views_swept": len(swept),
        "views_skipped": skipped,
        "planes_max": max((entry.get("planes") or 0) for entry in per_view) or None,
        "coverage_mean": round(float(np.mean(coverages)), 4) if coverages else 0.0,
        "coverage_median": round(float(np.median(coverages)), 4) if coverages else 0.0,
        "ncc_mean": round(float(np.mean(nccs)), 4) if nccs else 0.0,
        # The honest cross-check: dense vs the sparse points' own depth.
        "sparse_check": {
            "views": len(agreement_rel),
            "median_of_view_medians_rel_err":
                round(float(np.median(agreement_rel)), 4) if agreement_rel else None,
        },
        "cloud": cloud_stats,
        "parameters": {
            "dense_max_size": cfg.dense_max_size,
            "dense_neighbors": cfg.dense_neighbors,
            "dense_step_px": cfg.dense_step_px,
            "dense_max_planes": cfg.dense_max_planes,
            "dense_min_parallax": cfg.dense_min_parallax,
            "dense_block": cfg.dense_block,
            "dense_ncc_window": cfg.dense_ncc_window,
            "dense_min_ncc": cfg.dense_min_ncc,
            "cpu_only": True,
        },
        "per_view": per_view,
        "elapsed_seconds": round(time.time() - started, 2),
        "memory_samples": pressure,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    outputs = [meta_path] + sorted(cfg.d("dense").glob("*.npz"))
    if cloud_path is not None:
        outputs.append(cloud_path)
    ctx.add_output_files(outputs)
    ctx.log(
        f"dense sweep complete: {len(per_view)}/{len(views)} views, "
        f"coverage {meta['coverage_mean']:.0%}, "
        f"sparse agreement "
        f"{meta['sparse_check']['median_of_view_medians_rel_err']}, "
        f"{meta['elapsed_seconds']}s")
    ctx.result(meta)
    return meta
