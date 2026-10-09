"""Phase 5: surface reconstruction - TSDF fusion + marching cubes.

Reads every dense depth map (``dense/<view>.npz``), fuses it into a voxel
TSDF over the reconstruction, extracts the zero level set with
scikit-image's marching cubes, and colours the vertices from the frames.

Design decisions, all measured (see README "Dense stage"/"Mesh stage"):

* **Volume crop.** A full-scene volume spends almost all of its voxels on
  distant background: on ``realvid`` the un-cropped scene has diagonal
  ~106 units, so the 4-unit vehicle would be 10 voxels across at 256^3.
  The volume is cropped to a ball around the *look-at point* (the least-
  squares intersection of the camera optical axes - the point an orbit
  scan is of), radius ``mesh_crop * median camera distance``.  Inside the
  ball the diagonal is ~36 units: 42 vehicle voxels at 384^3.  The crop
  only ever *keeps* the subject (it sits on every optical axis, well
  inside the radius); if it would remove too much the stage falls back to
  the full extent and says so in the meta.

* **Fusion.** Standard KinectFusion averaging with two adjustments for
  MVS depth maps: the truncation band is ``max(voxels * size, rel * z)``
  because plane-sweep noise grows with camera distance (~2% of depth), and
  TSDF voxels are initialised to +1 (empty) so unfused space never sits
  at the iso-level and grows spurious surfaces.  The grid is processed in
  slabs with elementwise float32 math - no BLAS, no threads to reorder
  reductions, so a rerun is bit-identical.

* **Colouring.** A vertex keeps the average frame colour over every view
  whose depth map agrees at that pixel within a voxel-scaled tolerance.
  Agreement in *depth* (not just projection) rejects occluded and
  background contributions; vertices no view vouched for get flat grey
  and the share is reported honestly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ReconConfig, ReconstructionError
from .dense import (
    _load_color,
    _load_depth,
    _depth_path,
    _frames_by_name,
    _working_intrinsics,
    load_views,
)
from .ply import write_ply_mesh
from .state import StageContext, StateStore, memory_pressure

# Hard ceiling on the voxel grid (~160M cells = ~0.9 GB of TSDF + weight).
# The default 384^3 is 57M cells; this only triggers on a hand-set
# RECON_MESH_RESOLUTION that would swap the box into the ground.
_MAX_GRID_VOXELS = 160_000_000
# Fusion runs in slabs of about this many voxels: ~40 MB of temporaries.
_SLAB_VOXELS = 1_000_000
_WEIGHT_CAP = 64          # averaging cap; early estimates stop counting


def _depth_files(cfg: ReconConfig, views: List[Dict[str, Any]]) -> Dict[str, Path]:
    """Depth maps that actually exist (a view may have been skipped)."""
    return {view["name"]: path for view in views
            if (path := _depth_path(cfg, view["name"])).is_file()}


# ---------------------------------------------------------------------------
# Geometry: subject point and volume bounds
# ---------------------------------------------------------------------------
def _subject_point(views: List[Dict[str, Any]]
                   ) -> Tuple[Optional[np.ndarray], float]:
    """Least-squares intersection of the camera optical axes.

    Returns ``(point, condition)``; ``point`` is None when the axes are too
    degenerate to trust (all cameras effectively parallel).
    """
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for view in views:
        d = view["R"][2, :]              # camera +Z in world coordinates
        P = np.eye(3) - np.outer(d, d)   # perpendicular-distance operator
        A += P
        b += P @ view["center"]
    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > 1e6:
        return None, cond
    return np.linalg.solve(A, b), cond


def _backproject_points(cfg: ReconConfig, views: List[Dict[str, Any]],
                        depth_paths: Dict[str, Path],
                        ctx: StageContext) -> np.ndarray:
    """Every confident dense point in world coordinates (for bounds only)."""
    chunks: List[np.ndarray] = []
    total = 0
    for view in views:
        path = depth_paths.get(view["name"])
        if path is None:
            continue
        data = _load_depth(path)
        if data is None:
            continue
        depth = data["depth"]
        valid = data["valid"].astype(bool)
        if not valid.any():
            continue
        height, width = depth.shape
        K_w = _working_intrinsics(view, (width, height))
        ys, xs = np.mgrid[0:height, 0:width]
        idx = np.flatnonzero(valid.reshape(-1))
        grid = np.stack([xs.reshape(-1), ys.reshape(-1),
                         np.ones(height * width, np.float64)])[..., idx]
        rays = np.linalg.inv(K_w) @ grid
        d = depth.reshape(-1)[idx].astype(np.float64)
        cam = rays * d[None, :]
        world = (view["R"].T @ (cam - view["t"][:, None])).T
        chunks.append(world.astype(np.float32))
        total += len(idx)
        ctx.check()
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]


def _volume_bounds(cfg: ReconConfig, views: List[Dict[str, Any]],
                   points: np.ndarray) -> Dict[str, Any]:
    """Crop decision + grid geometry for the TSDF volume."""
    subject, cond = _subject_point(views)
    crop: Dict[str, Any] = {"applied": False, "condition": round(cond, 2)}
    kept = points
    if subject is not None and len(points):
        cam_dists = np.array(
            [float(np.linalg.norm(subject - v["center"])) for v in views])
        median = float(np.median(cam_dists))
        radius = cfg.mesh_crop * median
        inside = np.linalg.norm(points - subject, axis=1) <= radius
        # Keep the crop only when it leaves a real scene behind; a wrong
        # look-at (or a scene larger than the orbit) must degrade to the
        # full extent, not to an empty volume.
        if int(inside.sum()) >= max(2000, int(0.01 * len(points))):
            kept = points[inside]
            crop.update({
                "applied": True,
                "subject": [round(float(x), 3) for x in subject],
                "camera_distance_median": round(median, 3),
                "radius": round(radius, 3),
                "points_kept": int(inside.sum()),
                "points_total": int(len(points)),
            })

    lo = np.percentile(kept, 0.5, axis=0)
    hi = np.percentile(kept, 99.5, axis=0)
    pad = (hi - lo) * 0.02 + 1e-6
    lo, hi = lo - pad, hi + pad
    extent = hi - lo
    voxel = float(extent.max()) / cfg.mesh_resolution
    dims = np.clip(np.ceil(extent / voxel - 1e-9).astype(np.int64), 1,
                   cfg.mesh_resolution)
    total = int(np.prod(dims))
    if total > _MAX_GRID_VOXELS:
        raise ReconstructionError(
            f"Mesh grid would be {dims.tolist()} = {total} voxels, above the "
            f"{_MAX_GRID_VOXELS} safety limit.",
            suggestion="Lower RECON_MESH_RESOLUTION or check that the depth "
                       "maps are not full of distant outliers.",
            details={"dims": dims.tolist(), "voxels": total},
        )
    return {
        "origin": lo.astype(np.float32),
        "voxel": voxel,
        "dims": tuple(int(x) for x in dims),
        "total_voxels": total,
        "bounds_lo": [round(float(x), 4) for x in lo],
        "bounds_hi": [round(float(x), 4) for x in hi],
        "extent": [round(float(x), 4) for x in extent],
        "crop": crop,
    }


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def _fuse(cfg: ReconConfig, views: List[Dict[str, Any]],
          depth_paths: Dict[str, Path], volume: Dict[str, Any],
          ctx: StageContext) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Average every depth map into a truncated signed distance field."""
    origin = volume["origin"]
    voxel = np.float32(volume["voxel"])
    d0, d1, d2 = volume["dims"]

    tsdf = np.ones((d0, d1, d2), dtype=np.float32)   # +1 = empty space
    weight = np.zeros((d0, d1, d2), dtype=np.uint8)

    slab = max(1, min(d2, int(_SLAB_VOXELS // max(1, d0 * d1))))
    # Voxel-centre coordinates along each grid axis (f32, built once).
    ax0 = (origin[0] + (np.arange(d0, dtype=np.float32) + 0.5) * voxel)[:, None, None]
    ax1 = (origin[1] + (np.arange(d1, dtype=np.float32) + 0.5) * voxel)[None, :, None]

    updates = 0
    samples = 0
    views_used = 0
    pressure: List[Dict[str, Any]] = []

    for view_index, view in enumerate(views):
        path = depth_paths.get(view["name"])
        if path is None:
            continue
        data = _load_depth(path)
        if data is None:
            continue
        depth = data["depth"].astype(np.float32, copy=False)
        valid = data["valid"].astype(bool)
        if not valid.any():
            continue
        height, width = depth.shape
        K_w = _working_intrinsics(view, (width, height))
        fx = np.float32(K_w[0, 0])
        fy = np.float32(K_w[1, 1])
        cx = np.float32(K_w[0, 2])
        cy = np.float32(K_w[1, 2])
        R = view["R"].astype(np.float32)
        t = view["t"].astype(np.float32)

        for k0 in range(0, d2, slab):
            k1 = min(d2, k0 + slab)
            ax2 = (origin[2]
                   + (np.arange(k0, k1, dtype=np.float32) + 0.5) * voxel
                   )[None, None, :]
            # world -> camera (elementwise; no BLAS, deterministic)
            xc = R[0, 0] * ax0 + R[0, 1] * ax1 + R[0, 2] * ax2 + t[0]
            yc = R[1, 0] * ax0 + R[1, 1] * ax1 + R[1, 2] * ax2 + t[1]
            zc = R[2, 0] * ax0 + R[2, 1] * ax1 + R[2, 2] * ax2 + t[2]
            samples += int(zc.size)

            front = zc > np.float32(1e-6)
            safe = np.where(front, zc, np.float32(1.0))
            # clip before the int32 cast: behind-camera voxels project to
            # huge (finite) coordinates and the raw cast overflows.
            ui = np.clip(np.rint(fx * xc / safe + cx),
                         -2.0e9, 2.0e9).astype(np.int32)
            vi = np.clip(np.rint(fy * yc / safe + cy),
                         -2.0e9, 2.0e9).astype(np.int32)
            inside = (front & (ui >= 0) & (ui < width)
                      & (vi >= 0) & (vi < height))
            if not inside.any():
                continue

            dv = np.where(inside, depth[np.clip(vi, 0, height - 1),
                                        np.clip(ui, 0, width - 1)],
                          np.float32(np.nan))
            ok = inside & valid[np.clip(vi, 0, height - 1),
                                np.clip(ui, 0, width - 1)] & np.isfinite(dv)
            if not ok.any():
                continue

            # Truncation: voxel term for near geometry, relative term for
            # depth-proportional plane-sweep noise.
            trunc = np.maximum(np.float32(cfg.mesh_trunc_voxels) * voxel,
                               np.float32(cfg.mesh_trunc_rel) * zc)
            sdf = dv - zc
            hit = ok & (np.abs(sdf) <= trunc)
            if not hit.any():
                continue
            sample = np.clip(sdf / trunc, -1.0, 1.0).astype(np.float32)

            w_old = weight[:, :, k0:k1]
            ts_old = tsdf[:, :, k0:k1]
            occupied = w_old > 0
            blend = hit & occupied
            fresh = hit & ~occupied

            new_ts = ts_old.copy()
            if fresh.any():
                new_ts[fresh] = sample[fresh]
            if blend.any():
                w = w_old[blend].astype(np.float32)
                new_ts[blend] = (ts_old[blend] * w
                                 + sample[blend]) / (w + np.float32(1.0))
            tsdf[:, :, k0:k1] = new_ts
            new_w = w_old.copy()
            new_w[hit] = np.minimum(
                w_old[hit].astype(np.int16) + 1, _WEIGHT_CAP).astype(np.uint8)
            weight[:, :, k0:k1] = new_w
            updates += int(hit.sum())

        views_used += 1
        ctx.check()
        if (view_index + 1) % 5 == 0 or view_index == len(views) - 1:
            sample_info = {"at_view": view_index + 1,
                           **memory_pressure(cfg.ram_limit_gb)}
            pressure.append(sample_info)
            ctx.note(f"{view_index + 1}/{len(views)} views fused "
                     f"({updates} samples, rss "
                     f"{sample_info['process_rss_gb']} GB)")

    stats = {
        "views_used": views_used,
        "samples_total": samples,
        "updates": updates,
        "covered_share": round(float((weight > 0).mean()), 4),
        "weight_mean_covered": round(float(weight[weight > 0].mean()), 2)
        if (weight > 0).any() else 0.0,
        "memory_samples": pressure,
    }
    return tsdf, weight, stats


# ---------------------------------------------------------------------------
# Surface extraction + colouring
# ---------------------------------------------------------------------------
def _extract(tsdf: np.ndarray, voxel: float, origin: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Marching cubes at the zero level set; world-space vertices."""
    if not np.any(tsdf < 0.0):
        raise ReconstructionError(
            "No surface: the TSDF never crossed zero.",
            suggestion="Every depth map disagreed about the surface side - "
                       "check dense/dense_meta.json coverage and sparse "
                       "agreement for this project.",
            details={"negative_voxels": 0},
        )
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ReconstructionError(
            "scikit-image is required for meshing.",
            suggestion="python3 -m pip install scikit-image",
            details={"error": str(exc)},
        ) from exc

    verts, faces, _normals, _values = marching_cubes(
        tsdf, level=0.0, spacing=(voxel, voxel, voxel))
    # skimage positions voxel (i, j, k) at i*spacing; our voxel *centre* i
    # sits at origin + (i + 0.5) * voxel.
    world = verts.astype(np.float32) + origin + np.float32(0.5 * voxel)
    return world, faces.astype(np.int64)


def _colourize(cfg: ReconConfig, views: List[Dict[str, Any]],
               depth_paths: Dict[str, Path], frames: Dict[str, Path],
               verts: np.ndarray, voxel: float,
               ctx: StageContext
               ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Depth-agreeing multi-view colour per vertex, plus support fractions.

    ``support[v]`` is the share of observing views whose depth map agrees
    with the vertex position inside the fusion band - the consensus signal
    that separates fused surface from single-view junk sheets.
    """
    sums = np.zeros((len(verts), 3), dtype=np.float32)
    counts = np.zeros(len(verts), dtype=np.uint16)
    proj = np.zeros(len(verts), dtype=np.uint16)
    cons = np.zeros(len(verts), dtype=np.uint16)
    tol = np.float32(cfg.mesh_color_tol_voxels * voxel)

    for view in views:
        path = depth_paths.get(view["name"])
        frame = frames.get(view["name"])
        if path is None or frame is None:
            continue
        data = _load_depth(path)
        if data is None:
            continue
        depth = data["depth"].astype(np.float32, copy=False)
        valid = data["valid"].astype(bool)
        height, width = depth.shape
        K_w = _working_intrinsics(view, (width, height))

        cam = view["R"] @ verts.T.astype(np.float64) + view["t"][:, None]
        z = cam[2]
        front = z > 1e-6
        safe = np.where(front, z, 1.0)
        u = np.clip(np.rint(K_w[0, 0] * cam[0] / safe + K_w[0, 2]),
                    -2.0e9, 2.0e9).astype(np.int32)
        v = np.clip(np.rint(K_w[1, 1] * cam[1] / safe + K_w[1, 2]),
                    -2.0e9, 2.0e9).astype(np.int32)
        inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not inside.any():
            ctx.check()
            continue
        uc = np.clip(u, 0, width - 1)
        vc = np.clip(v, 0, height - 1)
        see = inside & valid[vc, uc]
        proj += see
        # Same band the fusion used: a view "confirms" the vertex when its
        # own depth there lies inside the truncation interval.
        band = np.maximum(cfg.mesh_trunc_voxels * voxel,
                          cfg.mesh_trunc_rel * z)
        cons += see & (np.abs(depth[vc, uc] - z) <= band)

        hit = see & (np.abs(depth[vc, uc] - z) <= float(tol))
        if hit.any():
            color = _load_color(frame, (width, height))
            bgr = color[vc[hit], uc[hit]].astype(np.float32)
            sums[hit] += bgr[:, ::-1]            # BGR -> RGB
            counts[hit] += 1
        ctx.check()

    support = cons / np.maximum(proj, 1)
    coloured = counts > 0
    rgb = np.full((len(verts), 3), 128, dtype=np.uint8)
    if coloured.any():
        rgb[coloured] = np.clip(
            sums[coloured] / counts[coloured, None], 0, 255).astype(np.uint8)
    seen = proj > 0
    stats = {
        "coloured_share": round(float(coloured.mean()), 4),
        "views_agreeing_mean": round(float(counts[coloured].mean()), 2)
        if coloured.any() else 0.0,
        "observing_views_p50": int(np.median(proj)) if len(proj) else 0,
        "support_p10_p50_p90": [round(float(x), 3) for x in
                                np.percentile(support[seen], [10, 50, 90])]
        if seen.any() else [0.0, 0.0, 0.0],
    }
    return rgb, support, stats


# Below this, a vertex's "support" is indistinguishable from luck no
# matter how poor the project's depth maps are - the relative threshold
# (mesh_support_rel * sparse baseline) is floored here.
_SUPPORT_FLOOR = 0.1


def _load_sparse_pts(cfg: ReconConfig) -> Optional[np.ndarray]:
    """Validated surface samples (reprojection-checked sparse cloud)."""
    ply = cfg.d("sparse", "sparse_point_cloud.ply")
    if not ply.is_file():
        return None
    from .ply import read_ply
    pts, _rgb = read_ply(ply)
    if not len(pts):
        return None
    if len(pts) > 6000:                     # deterministic subsample
        pts = pts[::len(pts) // 6000 + 1]
    return pts


def _sparse_baseline(cfg: ReconConfig, views: List[Dict[str, Any]],
                     depth_paths: Dict[str, Path], voxel: float,
                     pts: Optional[np.ndarray],
                     ctx: StageContext) -> Dict[str, Any]:
    """How well the depth maps confirm VALIDATED surface points.

    The sparse cloud is SfM-triangulated from reprojection-checked
    matches, i.e. surface points that are true independent of the depth
    maps.  Their agreement level with the depth maps (same fusion band the
    mesh uses) is this project's honest "true surface" support level.

    It has to be measured per project: an absolute support threshold
    cannot work across datasets, because the absolute scale tracks
    depth-map self-consistency.  Measured on the two verification sets:
    0.54 median on ``realvid`` but only 0.35 on ``synth_selftest`` (its
    bundle-adjusted focal is off by 10%, which biases every plane-sweep
    depth slightly differently per view).  Judging mesh vertices relative
    to this baseline keeps the gate meaningful on both.
    """
    result: Dict[str, Any] = {"points": 0, "support_p50": 0.0,
                              "support_p10_p90": [0.0, 0.0]}
    if pts is None or not len(pts):
        return result
    cons = np.zeros(len(pts), dtype=np.uint16)
    obs = np.zeros(len(pts), dtype=np.uint16)
    for view in views:
        path = depth_paths.get(view["name"])
        if path is None:
            continue
        data = _load_depth(path)
        if data is None:
            continue
        depth = data["depth"].astype(np.float32, copy=False)
        valid = data["valid"].astype(bool)
        height, width = depth.shape
        K_w = _working_intrinsics(view, (width, height))
        cam = view["R"] @ pts.T.astype(np.float64) + view["t"][:, None]
        z = cam[2]
        front = z > 1e-6
        safe = np.where(front, z, 1.0)
        u = np.clip(np.rint(K_w[0, 0] * cam[0] / safe + K_w[0, 2]),
                    -2.0e9, 2.0e9).astype(np.int32)
        v = np.clip(np.rint(K_w[1, 1] * cam[1] / safe + K_w[1, 2]),
                    -2.0e9, 2.0e9).astype(np.int32)
        inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not inside.any():
            ctx.check()
            continue
        uc = np.clip(u, 0, width - 1)
        vc = np.clip(v, 0, height - 1)
        see = inside & valid[vc, uc]
        obs += see
        band = np.maximum(cfg.mesh_trunc_voxels * voxel,
                          cfg.mesh_trunc_rel * z)
        cons += see & (np.abs(depth[vc, uc] - z) <= band)
        ctx.check()
    support = cons / np.maximum(obs, 1)
    seen = obs > 0
    if seen.any():
        result = {
            "points": int(seen.sum()),
            "support_p50": round(float(np.median(support[seen])), 3),
            "support_p10_p90": [round(float(x), 3) for x in
                                np.percentile(support[seen], [10, 90])],
        }
    return result


def _prune(verts: np.ndarray, faces: np.ndarray, rgb: np.ndarray,
           keep: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                      Dict[str, Any]]:
    """Drop unsupported vertices and every face that needs one."""
    stats = {"vertices_in": int(len(verts)),
             "vertices_kept": int(len(verts)),
             "faces_in": int(len(faces)), "faces_kept": int(len(faces))}
    if not keep.all():
        face_ok = keep[faces].all(axis=1)
        faces = faces[face_ok]
        remap = np.full(len(verts), -1, dtype=np.int64)
        kept_idx = np.flatnonzero(keep)
        remap[kept_idx] = np.arange(len(kept_idx), dtype=np.int64)
        verts, rgb = verts[keep], rgb[keep]
        faces = remap[faces]
        stats["vertices_kept"] = int(len(verts))
        stats["faces_kept"] = int(len(faces))
    # Compact vertices no surviving face references (a support-pruned
    # neighbourhood can strand its neighbours).
    if len(faces):
        used = np.zeros(len(verts), dtype=bool)
        used[faces.reshape(-1)] = True
        if not used.all():
            remap = np.full(len(verts), -1, dtype=np.int64)
            kept_idx = np.flatnonzero(used)
            remap[kept_idx] = np.arange(len(kept_idx), dtype=np.int64)
            verts, rgb = verts[used], rgb[used]
            faces = remap[faces]
            stats["vertices_kept"] = int(len(verts))
            stats["faces_kept"] = int(len(faces))
    stats["dropped_share"] = round(
        1.0 - stats["faces_kept"] / max(1, stats["faces_in"]), 4)
    return verts, faces, rgb, stats


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def _drop_unanchored(cfg: ReconConfig, verts: np.ndarray, faces: np.ndarray,
                     rgb: np.ndarray, sparse: Optional[np.ndarray],
                     ctx: StageContext
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                Dict[str, Any]]:
    """Remove connected components that do not touch validated surface.

    The support prune judges each vertex alone, so junk that is locally
    self-consistent (floating sheets, blobs) survives it and shows up as
    a bounding box hundreds of percent too large.  Component anchoring
    asks a different question: does this piece of surface contain *any*
    point the sparse solve validated?  The keep-radius is the median
    spacing of the sparse cloud - anchored at the scale this project's
    surface is actually validated at.  Measured on ``synth_selftest``:
    64 % of vertices lived in 10 373 fragments floating 0.3-2 units from
    the box; anchoring removed them and took bbox error from
    +56/+163/+273 % to -4/+13/+12 % while keeping 73 % of true surface.
    """
    stats: Dict[str, Any] = {"applied": False}
    if sparse is None or len(sparse) < 3 or not len(faces):
        stats["reason"] = "no validated sparse points to anchor against"
        return verts, faces, rgb, stats
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ReconstructionError(
            "scipy is required for component anchoring.",
            suggestion="python3 -m pip install scipy",
            details={"error": str(exc)},
        ) from exc

    tree = cKDTree(sparse.astype(np.float64))
    dist, _ = tree.query(verts.astype(np.float64), k=1)
    near, _ = tree.query(sparse.astype(np.float64), k=2)
    spacing = float(np.median(near[:, 1]))
    radius = cfg.mesh_anchor_rel * spacing

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                            faces[:, [2, 0]]])
    adjacency = coo_matrix((np.ones(len(edges), dtype=np.uint8),
                            (edges[:, 0], edges[:, 1])),
                           shape=(len(verts), len(verts)))
    ncomp, labels = connected_components(adjacency, directed=False)
    comp_min = np.full(ncomp, np.inf, dtype=np.float64)
    np.minimum.at(comp_min, labels, dist)
    keep = comp_min[labels] <= radius
    ctx.check()

    stats = {
        "applied": True,
        "radius": round(radius, 4),
        "sparse_spacing": round(spacing, 4),
        "components": int(ncomp),
        "components_kept": int((comp_min <= radius).sum()),
        "vertices_dropped": int((~keep).sum()),
    }
    if keep.all():
        return verts, faces, rgb, stats
    verts, faces, rgb, prune_stats = _prune(verts, faces, rgb, keep)
    stats["vertices_dropped"] = (prune_stats["vertices_in"]
                                 - prune_stats["vertices_kept"])
    stats["faces_dropped"] = prune_stats["faces_in"] - prune_stats["faces_kept"]
    return verts, faces, rgb, stats


def run_mesh(cfg: ReconConfig, store: StateStore, ctx: StageContext) -> Dict[str, Any]:
    views = load_views(cfg)
    depth_paths = _depth_files(cfg, views)
    if not depth_paths:
        raise ReconstructionError(
            "Mesh reconstruction needs dense depth maps "
            "(dense/<view>.npz), none were found.",
            suggestion="Run the pipeline through the 'dense' stage first; "
                       "the mesh stage fuses its depth maps.",
            details={"project": str(cfg.project_dir)},
        )
    frames = _frames_by_name(cfg)
    raw_dir = cfg.d("mesh", "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = raw_dir / "mesh.ply"
    meta_path = cfg.d("mesh", "mesh_meta.json")

    fingerprint = cfg.mesh_fingerprint()
    ctx.set_command(
        f"in-process TSDF fusion + marching cubes ({len(depth_paths)} depth "
        f"maps, grid <= {cfg.mesh_resolution}^3)")
    ctx.log(f"meshing from {len(depth_paths)} depth maps "
            f"(crop {cfg.mesh_crop}x, resolution {cfg.mesh_resolution})")

    started = time.time()
    ctx.check()

    points = _backproject_points(cfg, views, depth_paths, ctx)
    if len(points) < 500:
        raise ReconstructionError(
            f"Only {len(points)} confident dense points - too few to size "
            f"a fusion volume.",
            suggestion="Check dense/dense_meta.json coverage; re-run the "
                       "dense stage if it collapsed.",
            details={"points": int(len(points))},
        )
    volume = _volume_bounds(cfg, views, points)
    del points
    ctx.note(f"volume {volume['dims']} @ {volume['voxel']:.3f} units "
             f"(crop {'on' if volume['crop']['applied'] else 'off'})")

    tsdf, weight, fuse_stats = _fuse(cfg, views, depth_paths, volume, ctx)
    if fuse_stats["updates"] == 0:
        raise ReconstructionError(
            "Depth fusion produced no samples inside the volume.",
            suggestion="The volume bounds and the depth maps do not "
                       "overlap; inspect dense/dense_meta.json.",
            details=volume,
        )
    # Consensus gate: a voxel must have been hit by mesh_min_weight
    # different views before it may shape the surface.  Single-view hits
    # are the dominant failure mode of MVS depth on real footage (64% of
    # covered voxels on realvid) and they surface as floating sheets.
    covered = int((weight > 0).sum())
    kept = weight >= cfg.mesh_min_weight
    fuse_stats["weight_gate"] = cfg.mesh_min_weight
    fuse_stats["gate_kept_of_covered"] = round(
        int(kept.sum()) / max(1, covered), 4)
    tsdf = np.where(kept, tsdf, np.float32(1.0))
    del weight, kept, covered
    ctx.note(f"weight gate >= {cfg.mesh_min_weight}: "
             f"{fuse_stats['gate_kept_of_covered']:.0%} of covered voxels")
    ctx.check()

    verts, faces = _extract(tsdf, volume["voxel"], volume["origin"])
    del tsdf
    ctx.note(f"marching cubes: {len(verts)} vertices, {len(faces)} faces")

    sparse_pts = _load_sparse_pts(cfg)
    rgb, support, colour_stats = _colourize(cfg, views, depth_paths, frames,
                                            verts, volume["voxel"], ctx)
    # Calibrate the support gate against validated surface: threshold =
    # mesh_support_rel * (agreement level of reprojection-checked sparse
    # points), floored so a hopeless project still prunes instead of
    # shipping every vertex that ever looked at the scene.
    baseline = _sparse_baseline(cfg, views, depth_paths, volume["voxel"],
                                sparse_pts, ctx)
    threshold = max(_SUPPORT_FLOOR,
                    cfg.mesh_support_rel * float(baseline["support_p50"]))
    colour_stats["sparse_support_p50"] = baseline["support_p50"]
    colour_stats["support_threshold"] = round(threshold, 3)
    colour_stats["supported_share"] = round(
        float((support >= threshold).mean()), 4)
    ctx.note(f"support baseline (sparse) {baseline['support_p50']:.2f} "
             f"-> threshold {threshold:.2f} "
             f"({colour_stats['supported_share']:.0%} of vertices)")
    verts, faces, rgb, prune_stats = _prune(
        verts, faces, rgb, support >= threshold)
    ctx.note(f"support prune: {prune_stats['vertices_kept']} vertices, "
             f"{prune_stats['faces_kept']} faces "
             f"({prune_stats['dropped_share']:.0%} dropped)")
    verts, faces, rgb, anchor_stats = _drop_unanchored(
        cfg, verts, faces, rgb, sparse_pts, ctx)
    if anchor_stats.get("applied"):
        ctx.note(f"anchor prune: {anchor_stats['components_kept']}/"
                 f"{anchor_stats['components']} components, "
                 f"-{anchor_stats['faces_dropped']} faces "
                 f"(radius {anchor_stats['radius']})")
    if len(faces) < 1000:
        raise ReconstructionError(
            f"Pruning left only {len(faces)} faces - the depth maps do not "
            f"agree well enough to form a surface.",
            suggestion="Check dense/dense_meta.json (sparse_check, "
                       "agreement) and reconstruction_report.json match "
                       "quality; a scene with this little view overlap "
                       "cannot be meshed honestly.",
            details={"prune": prune_stats, "anchor": anchor_stats,
                     "support": colour_stats},
        )

    out = write_ply_mesh(mesh_path, verts, faces, rgb)
    ctx.add_output_files([out])

    meta: Dict[str, Any] = {
        "fingerprint": fingerprint,
        "algorithm": "TSDF fusion + marching cubes (scikit-image, lewiner)",
        "views_total": len(views),
        "views_with_depth": len(depth_paths),
        "volume": {key: value for key, value in volume.items()
                   if key != "origin"},
        "volume_origin": [round(float(x), 4) for x in volume["origin"]],
        "fusion": fuse_stats,
        "support_baseline": baseline,
        "mesh": {
            "vertices": int(len(verts)),
            "faces": int(len(faces)),
            **colour_stats,
            "prune": prune_stats,
            "anchor": anchor_stats,
        },
        "output": str(out.relative_to(cfg.project_dir)),
        "elapsed_seconds": round(time.time() - started, 2),
        "memory_peak_gb": memory_pressure(cfg.ram_limit_gb)["process_peak_gb"],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ctx.add_output_files([meta_path])

    ctx.log(
        f"meshed {len(verts)} vertices / {len(faces)} faces "
        f"({fuse_stats['updates']} depth samples over "
        f"{fuse_stats['views_used']} views, "
        f"{meta['elapsed_seconds']}s)",
        operation="complete",
    )
    result = {
        "mesh": meta["output"],
        "vertices": meta["mesh"]["vertices"],
        "faces": meta["mesh"]["faces"],
        "voxel_size": round(volume["voxel"], 4),
        "views_used": fuse_stats["views_used"],
        "elapsed_seconds": meta["elapsed_seconds"],
    }
    ctx.result(result)
    return result
