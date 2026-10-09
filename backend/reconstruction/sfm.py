"""PHASE 3 - incremental Structure-from-Motion on CPU.

Steps implemented here (sections 10-12 and 31):

    verified matches -> feature tracks -> seed camera pair (essential matrix)
                     -> triangulate initial points
                     -> register further cameras (PnP) one by one
                     -> triangulate new points
                     -> local bundle adjustment after each registration
                     -> final bundle adjustment
                     -> validation report + sparse point cloud export

Everything is numpy/SciPy/OpenCV on the CPU.  Bundle adjustment uses a
sparse Jacobian (SciPy ``least_squares`` with ``jac_sparsity``), which
keeps memory flat on 6 GB and, thanks to graph coloring, needs only a
handful of residual evaluations per iteration.

Honesty rules (sections 6, 10, 12): a camera that cannot be registered is
reported, not hidden; a disconnected camera graph is reported with numbers;
missing view directions are reported ("Rear geometry insufficiently
observed.") instead of being invented.
"""

from __future__ import annotations

import json
import dataclasses
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import ReconConfig, ReconstructionError
from .features import load_keypoints_only, selected_frames
from .matching import assumed_intrinsics, list_match_files, load_match
from .state import StateStore, StageContext, memory_pressure

SECTOR_NAMES = (
    "front", "front-left", "left", "rear-left",
    "rear", "rear-right", "right", "front-right",
)


def _cv2():
    import cv2
    return cv2


def _np():
    import numpy as np
    return np


# ---------------------------------------------------------------------------
# Feature tracks
# ---------------------------------------------------------------------------
def build_tracks(matches: List[Tuple[int, int, Dict[str, Any]]]
                  ) -> Tuple[List[List[Tuple[int, int]]], Dict[int, int]]:
    """Union verified matches into tracks: one 3D point per track.

    Returns (tracks, lookup) where lookup maps (image<<32 | kp) -> track id.
    """
    np = _np()
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != x:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j, data in matches:
        for a, b in zip(data["idx_a"].tolist(), data["idx_b"].tolist()):
            na = (int(i) << 32) | int(a)
            nb = (int(j) << 32) | int(b)
            parent.setdefault(na, na)
            parent.setdefault(nb, nb)
            union(na, nb)

    grouped: Dict[int, List[int]] = defaultdict(list)
    for node in parent:
        grouped[find(node)].append(node)

    tracks: List[List[Tuple[int, int]]] = []
    lookup: Dict[int, int] = {}
    for root in sorted(grouped):
        observations = sorted((int(node >> 32), int(node & 0xFFFFFFFF))
                              for node in grouped[root])
        # A track must be observed twice to constrain anything.
        if len(observations) < 2:
            continue
        tid = len(tracks)
        tracks.append(observations)
        for img, kp in observations:
            lookup[(img << 32) | kp] = tid
    return tracks, lookup


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def camera_center(R, t):
    """World-space position of a camera with world->cam pose (R, t)."""
    np = _np()
    return -R.T @ np.asarray(t, dtype=np.float64).reshape(3)


def _positive_depth(view, X) -> bool:
    """True when the point sits in front of the camera (chirality)."""
    np = _np()
    cam = np.asarray(view["R"], dtype=np.float64) @ \
        np.asarray(X, dtype=np.float64).reshape(3) + \
        np.asarray(view["t"], dtype=np.float64).reshape(3)
    return bool(cam[2] > 0)


def projection_matrix(K, R, t):
    np = _np()
    Rt = np.hstack([np.asarray(R, dtype=np.float64),
                    np.asarray(t, dtype=np.float64).reshape(3, 1)])
    return np.asarray(K, dtype=np.float64) @ Rt


def triangulate_two(view1, view2, pts1, pts2):
    """Triangulate N correspondences; returns (X (N,3), valid mask)."""
    cv2, np = _cv2(), _np()
    P1 = projection_matrix(view1["K"], view1["R"], view1["t"])
    P2 = projection_matrix(view2["K"], view2["R"], view2["t"])
    homogeneous = cv2.triangulatePoints(
        P1, P2,
        np.float64(pts1).T.copy(), np.float64(pts2).T.copy(),
    )
    w = homogeneous[3]
    X = (homogeneous[:3] / np.where(np.abs(w) < 1e-12, 1e-12, w)).T
    valid = np.isfinite(X).all(axis=1) & (np.abs(w) > 1e-12)
    return X, valid


def _dump_pnp(cfg, name, obj_pts, img_pts, K) -> None:
    """RECON_DEBUG_PNP=1 writes the failing correspondence set for analysis."""
    import os
    if not os.getenv("RECON_DEBUG_PNP"):
        return
    np = _np()
    try:
        np.savez_compressed(
            cfg.d("logs", f"debug_pnp_{name}.npz"),
            obj=np.float64(obj_pts), img=np.float64(img_pts),
            K=np.float64(K))
    except OSError:
        pass


def _pnp_diagnostic(obj_pts, img_pts, K) -> str:
    """Short, human-readable hint when PnP finds no consensus pose.

    Two usual culprits are worth separating in the log: a wrong initial
    focal length (the poses were built from an assumed field of view) and a
    threshold that is tighter than the model actually achieves.  Reporting
    RANSAC inlier counts at several thresholds turns a dead-end "no
    consensus" into an actionable message.
    """
    np = _np()
    cv2 = _cv2()
    hints = []
    for threshold in (3.0, 6.0, 12.0):
        try:
            ok, _, _, inl = cv2.solvePnPRansac(
                np.float64(obj_pts).reshape(-1, 1, 3),
                np.float64(img_pts).reshape(-1, 1, 2),
                K, None, iterationsCount=300, reprojectionError=threshold,
                confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            hints.append(f"@{threshold:g}px: solver error")
            continue
        count = 0 if inl is None else len(np.asarray(inl).ravel())
        hints.append(f"@{threshold:g}px: {count if ok else 0}/{len(obj_pts)}")
    return " [RANSAC inliers " + ", ".join(hints) + "]"


def project_points(X, rvec, tvec, K):
    """cv2.projectPoints wrapper (OpenCV 5 requires distCoeffs explicitly).

    Frames are assumed pre-undistorted: video/photo lens distortion is small
    at this scale and BA absorbs what remains.
    """
    np = _np()
    cv2 = _cv2()
    projected, _ = cv2.projectPoints(
        np.float64(X).reshape(-1, 3),
        np.float64(rvec).reshape(3, 1),
        np.float64(tvec).reshape(3, 1),
        np.asarray(K, dtype=np.float64),
        None,
    )
    return projected.reshape(-1, 2)


def reprojection_errors(view, X, pts):
    np = _np()
    if len(X) == 0:
        return np.zeros(0)
    projected = project_points(X, view["rvec"], view["t"], view["K"])
    return np.linalg.norm(projected - np.float64(pts), axis=1)


def parallax_degrees(C1, C2, X):
    np = _np()
    v1 = np.asarray(X) - np.asarray(C1)
    v2 = np.asarray(X) - np.asarray(C2)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    good = (n1 > 1e-9) & (n2 > 1e-9)
    cos = np.sum(v1 * v2, axis=-1) / np.where(good, n1 * n2, 1.0)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


# ---------------------------------------------------------------------------
# Match graph
# ---------------------------------------------------------------------------
def match_components(n_images: int, pairs) -> Tuple[Any, Dict[int, int]]:
    """Connected components of the verified-match graph.

    Returns ``(find, sizes)`` where ``find(i)`` is the component root of
    image ``i`` and ``sizes[root]`` is how many images share it.  This graph
    is decisive: registration can only ever grow *inside* the component the
    seed starts in, so an image in a component with no seed can never be
    registered no matter how good its own matches are.
    """
    parent = list(range(n_images))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for item in pairs:
        union(int(item[0]), int(item[1]))
    sizes: Dict[int, int] = defaultdict(int)
    for i in range(n_images):
        sizes[find(i)] += 1
    return find, sizes


# ---------------------------------------------------------------------------
# Bundle adjustment
# ---------------------------------------------------------------------------
def bundle_adjust(
    views: Dict[int, Dict[str, Any]],
    points: "Any",           # (M,3) float64 array, modified in place
    obs_cam: "Any",          # (R,) int
    obs_pt: "Any",           # (R,) int
    obs_xy: "Any",           # (R,2) float64
    *,
    camera_ids: Sequence[int],
    point_ids: Sequence[int],
    anchor_id: Optional[int],
    refine_f: bool,
    fallback_f: float,
    max_nfev: int = 60,
    f_prior_rel: float = 0.05,
    all_observers: bool = False,
) -> Dict[str, Any]:
    """Sparse, robust bundle adjustment over a subset of cameras/points.

    Parameters per optimized camera: axis-angle rotation + translation.
    Parameters per optimized point: XYZ.  Optionally one shared focal
    length with a Gaussian prior of ``f_prior_rel`` (relative sigma) so it
    can move away from the initial guess but never run away.

    With ``all_observers`` every observation of the selected points is kept
    in the objective while only ``camera_ids`` are optimized: a point seen
    by cameras outside a local window must keep satisfying them, otherwise
    the window refits the shared structure to itself and the global model
    drifts apart (the local window is then actively harmful).
    """
    np = _np()
    cv2 = _cv2()
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix

    cams = [c for c in sorted(camera_ids) if c in views]
    opt_cams = [c for c in cams if c != anchor_id]
    opt_pts = [int(p) for p in sorted(int(p) for p in point_ids)]
    if not opt_cams and not opt_pts:
        return {"performed": False, "reason": "nothing to optimize"}

    # Observations that touch the optimized subset.
    cam_index = {c: slot for slot, c in enumerate(opt_cams)}
    pt_index = {p: slot for slot, p in enumerate(opt_pts)}
    if all_observers:
        cam_ok = np.ones(len(obs_cam), dtype=bool)
    else:
        cam_ok = np.isin(obs_cam, np.array(cams, dtype=np.int64))
    pt_ok = np.isin(obs_pt, np.array(opt_pts, dtype=np.int64))
    sel = np.nonzero(cam_ok & pt_ok)[0]
    if len(sel) < 30:
        return {"performed": False, "reason": f"only {len(sel)} observations"}

    o_cam = obs_cam[sel]
    o_pt = obs_pt[sel]
    o_xy = obs_xy[sel]
    n_obs = len(sel)
    # Cameras that constrain the problem but are not being optimized stay at
    # their current pose (the residual reads them from ``views``).
    used_cams = sorted({int(c) for c in np.unique(o_cam)})

    def mean_reprojection() -> float:
        """Mean pixel error of exactly the observations this solve covers."""
        chunks = []
        for c in used_cams:
            m = o_cam == c
            if not m.any():
                continue
            proj = project_points(points[o_pt[m]],
                                  views[c]["rvec"], views[c]["t"],
                                  views[c]["K"])
            chunks.append(np.linalg.norm(np.asarray(proj) - o_xy[m], axis=1))
        if not chunks:
            return float("nan")
        return float(np.concatenate(chunks).mean())

    # --- parameter vector ------------------------------------------------
    base_f = float(views[cams[0]]["f"])
    pt_rows = np.array(opt_pts, dtype=np.int64)
    x0: List[float] = []
    for c in opt_cams:
        rvec, _ = cv2.Rodrigues(np.asarray(views[c]["R"], dtype=np.float64))
        x0.extend(rvec.ravel().tolist())
        x0.extend(np.asarray(views[c]["t"], dtype=np.float64).ravel().tolist())
    pt_start = len(x0)
    x0.extend(np.asarray(points[pt_rows], dtype=np.float64).ravel().tolist())
    if refine_f:
        x0.append(base_f)

    n_pose = len(opt_cams) * 6
    n_pts = len(opt_pts) * 3

    def unpack(x):
        poses = {}
        for slot, c in enumerate(opt_cams):
            chunk = x[slot * 6:slot * 6 + 6]
            # Keep the axis-angle vector: projectPoints takes rvec directly.
            poses[c] = (np.float64(chunk[:3]), np.float64(chunk[3:6]))
        pts = np.asarray(x[pt_start:pt_start + n_pts],
                         dtype=np.float64).reshape(-1, 3)
        f = float(x[-1]) if refine_f else base_f
        return poses, pts, f

    residual_len = n_obs * 2 + (1 if refine_f else 0)

    def _points_for(pt_ids):
        """(N,3) points for the given point rows (all are optimized here)."""
        slots = np.fromiter((pt_index[int(p)] for p in pt_ids),
                            dtype=np.int64, count=len(pt_ids))
        return pts_cache[slots].reshape(-1, 1, 3)

    pts_cache = np.asarray(points[pt_rows], dtype=np.float64) if len(pt_rows) \
        else np.zeros((0, 3))

    def residual(x) -> "Any":
        nonlocal pts_cache
        poses, pts, f = unpack(x)
        pts_cache = pts
        out = np.empty(residual_len, dtype=np.float64)
        if refine_f:
            sigma = f_prior_rel * base_f or 1.0
            out[-1] = (f - base_f) / sigma      # Gaussian prior on focal
        # One projectPoints call per camera keeps the inner loop vectorised.
        for c in used_cams:
            mask = o_cam == c
            if not mask.any():
                continue
            if c in poses:
                rvec, t = poses[c]
            else:                       # anchor camera stays fixed
                rvec, t = views[c]["rvec"], views[c]["t"]
            idx = np.nonzero(mask)[0]
            K = np.asarray(views[c]["K"], dtype=np.float64)
            if refine_f:
                K = K.copy()
                K[0, 0] = f
                K[1, 1] = f
            proj = project_points(_points_for(o_pt[idx]).reshape(-1, 3),
                                  rvec, t, K)
            out[idx * 2] = proj[:, 0] - o_xy[idx, 0]
            out[idx * 2 + 1] = proj[:, 1] - o_xy[idx, 1]
        return out

    # --- sparsity pattern: each residual row touches its camera (6) and
    # its point (3) parameters.  Graph coloring later reduces the numerical
    # Jacobian to a handful of residual evaluations per iteration.  The
    # anchor camera has no parameters, so its rows only carry point columns.
    # The shared focal length touches every projection row as well: leaving
    # it out of the pattern would tell the solver that focal only moves the
    # prior residual, and it would never leave the initial guess.
    col_rows: List[int] = []
    col_cols: List[int] = []
    focal_col = len(x0) - 1 if refine_f else -1
    for r in range(n_obs):
        cam = int(o_cam[r])
        base_col_pt = pt_start + pt_index[int(o_pt[r])] * 3
        for dof in (0, 1):                      # x and y residuals
            row = 2 * r + dof
            if cam in cam_index:
                base_col_cam = cam_index[cam] * 6
                for offset in range(6):         # rotation + translation
                    col_rows.append(row)
                    col_cols.append(base_col_cam + offset)
            for offset in range(3):             # point XYZ
                col_rows.append(row)
                col_cols.append(base_col_pt + offset)
            if refine_f:                        # shared focal
                col_rows.append(row)
                col_cols.append(focal_col)
    if refine_f:
        col_rows.append(2 * n_obs)
        col_cols.append(focal_col)
    n_rows = n_obs * 2 + (1 if refine_f else 0)
    n_cols = len(x0)
    jac_sparsity = coo_matrix(
        (np.ones(len(col_rows)), (np.array(col_rows, dtype=np.int64),
                                  np.array(col_cols, dtype=np.int64))),
        shape=(n_rows, n_cols)).tocsr()

    try:
        # ftol/xtol are *relative* stopping tests, and 1e-4 fires after the
        # first couple of trust-region steps: the solver quits while the
        # gradient is still enormous (measured optimality ~5.7e3) and the
        # model barely moves - a silent no-op bundle adjustment.  Measured on
        # a synthetic scene: 1e-4 stops at 2.37 px after 14 evaluations,
        # 1e-8 converges to the 0.5 px noise floor in 35.  Do not tighten
        # back: going below ~1e-12 makes trf divide by a collapsed radius.
        solution = least_squares(
            residual, np.float64(x0),
            jac_sparsity=jac_sparsity,
            method="trf", loss="huber", f_scale=1.5,
            x_scale="jac", ftol=1e-8, xtol=1e-8, max_nfev=max_nfev,
            verbose=0,
        )
    except Exception as exc:  # noqa: BLE001 - BA must never kill the run
        return {"performed": False, "reason": f"solver error: {exc}"}

    if not solution.success and solution.status < 0:
        return {"performed": False,
                "reason": f"solver stopped: {solution.message}"}

    # Honest before/after: the robust loss makes scipy's own cost a poor
    # progress signal (and reporting it twice as "initial" and "final" was
    # simply wrong), so measure what a human actually asks - how far the
    # observations sit from the model, in pixels, before and after the solve.
    mean_before = mean_reprojection()
    poses, pts, f = unpack(solution.x)
    for c, (rvec, t) in poses.items():
        R, _ = cv2.Rodrigues(rvec)
        views[c]["R"] = R
        views[c]["t"] = t
        views[c]["rvec"] = np.float64(rvec).ravel()
    for c in used_cams:              # shared focal reaches every camera
        views[c]["f"] = f
        K = np.asarray(views[c]["K"], dtype=np.float64).copy()
        K[0, 0] = f
        K[1, 1] = f
        views[c]["K"] = K
    if len(opt_pts):
        points[np.array(opt_pts, dtype=np.int64)] = pts
    mean_after = mean_reprojection()
    # How far the solve actually moved.  Equal before/after means on their own
    # are ambiguous - "already at the robust-loss optimum" and "the solver
    # never moved" look identical from the residual alone - so record the
    # displacement that tells them apart.
    x0_arr = np.asarray(x0, dtype=np.float64)
    parameter_shift = (float(np.max(np.abs(solution.x - x0_arr)))
                       if x0_arr.size else 0.0)
    return {
        "performed": True,
        "cameras": len(opt_cams),
        "points": len(opt_pts),
        "observations": int(n_obs),
        "iterations": int(solution.nfev),
        "mean_reprojection_before_px": round(mean_before, 4),
        "mean_reprojection_after_px": round(mean_after, 4),
        "parameter_shift": round(parameter_shift, 9),
        "cost": float(solution.cost),
        # Why the solver stopped matters when a refit looks like a no-op:
        # status 3 (xtol) with a large optimality means it quit early.
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "refine_focal": bool(refine_f),
        "focal": float(f),
    }


def calibrate_focal(
    views: Dict[int, Dict[str, Any]],
    points: "Any",
    obs_cam: "Any",
    obs_pt: "Any",
    obs_xy: "Any",
    *,
    camera_ids: Sequence[int],
    point_ids: Sequence[int],
    anchor_id: Optional[int],
    cfg: "ReconConfig",
) -> Dict[str, Any]:
    """Find the shared focal length by continuation over a profile.

    Two traps make the obvious implementations wrong.  Optimising focal
    jointly with poses and structure is nearly flat - the points absorb the
    error and the solver never leaves the initial guess - and a *wide* grid
    re-optimised from one state always reports that state's own focal as
    best, because a jump of 20% leaves the basin the model currently sits
    in.  So one small step is probed at a time, poses and structure are
    re-optimised after every accepted step, and only then is the result
    refined jointly (sections 7 and 12).
    """
    np = _np()
    cams = [c for c in sorted(camera_ids) if c in views]
    if not cams:
        return {"performed": False, "reason": "no cameras"}
    base_f = float(views[cams[0]]["f"])

    pt_rows = np.array(sorted({int(p) for p in point_ids}), dtype=np.int64)

    def state_of():
        snap = {c: {k: (np.array(v) if isinstance(v, np.ndarray) else v)
                    for k, v in views[c].items()} for c in cams}
        pts = np.array(points[pt_rows], dtype=np.float64) if len(pt_rows) \
            else None
        return snap, pts

    def restore_state(snap, pts) -> None:
        for c, values in snap.items():
            views[c].update(values)
        if pts is not None and len(pt_rows):
            points[pt_rows] = pts

    def apply_focal(f: float) -> None:
        for c in cams:
            K = np.asarray(views[c]["K"], dtype=np.float64).copy()
            K[0, 0] = f
            K[1, 1] = f
            views[c]["K"] = K
            views[c]["f"] = f

    def evaluate(f: float, snap, pts, nfev: int):
        """Restore a state, set the focal length, re-optimise, return cost."""
        restore_state(snap, pts)
        apply_focal(f)
        info = bundle_adjust(
            views, points, obs_cam, obs_pt, obs_xy,
            camera_ids=cams, point_ids=point_ids, anchor_id=anchor_id,
            refine_f=False, fallback_f=f, max_nfev=nfev,
        )
        if info.get("performed") and info.get("cost") is not None:
            return float(info["cost"])
        return None

    probe_nfev = max(20, cfg.ba_max_nfev // 2)
    start_snap, start_pts = state_of()
    start_cost = evaluate(base_f, start_snap, start_pts, probe_nfev)
    if start_cost is None:
        restore_state(start_snap, start_pts)
        return {"performed": False, "reason": "bundle adjustment did not run"}

    best_f, best_cost = float(base_f), start_cost
    best_state = state_of()
    tried: List[Dict[str, Any]] = [
        {"focal": best_f, "cost": best_cost, "step": 0.0, "accepted": True}]

    # Small steps first: the model has to be carried along the valley.
    steps = (0.03, -0.03, 0.06, -0.06, 0.12, -0.12, 0.25, -0.25)
    seen = {best_f}
    for _sweep in range(3):
        advanced = False
        for step in steps:
            f = float(best_f * (1.0 + step))
            if f < 0.5 * base_f or f > 2.0 * base_f:
                continue
            if any(abs(f - seen_f) < 1e-6 for seen_f in seen):
                continue
            seen.add(f)
            cost = evaluate(f, best_state[0], best_state[1], probe_nfev)
            entry = {"focal": f, "cost": cost, "step": step,
                     "accepted": False}
            if cost is not None and cost < best_cost * (1.0 - 1e-4):
                entry["accepted"] = True
                best_f, best_cost, best_state = f, cost, state_of()
                advanced = True
            tried.append(entry)
            if advanced:
                break          # greedy: keep walking from the new state
        if not advanced:
            break

    # Full budget at the winner, then let the joint refinement move freely
    # in a small neighbourhood of it.
    final_cost = evaluate(best_f, best_state[0], best_state[1],
                          cfg.ba_max_nfev)
    if final_cost is not None:
        best_cost = final_cost
    refine = bundle_adjust(
        views, points, obs_cam, obs_pt, obs_xy,
        camera_ids=cams, point_ids=point_ids, anchor_id=anchor_id,
        refine_f=bool(cfg.ba_refine_intrinsics), fallback_f=best_f,
        max_nfev=cfg.ba_max_nfev, f_prior_rel=cfg.ba_focal_prior,
    )
    return {
        "performed": True,
        "focal_initial": float(base_f),
        "focal": float(views[cams[0]]["f"]),
        "cost_at_initial_guess": start_cost,
        "best_walk_focal": float(best_f),
        "best_walk_cost": best_cost,
        "moved": bool(abs(best_f - base_f) > 1e-6),
        "refine": refine,
        "steps": tried,
    }


# ---------------------------------------------------------------------------
# Coverage analysis (section 6)
# ---------------------------------------------------------------------------
def analyze_coverage(cfg: ReconConfig, views: Dict[int, Dict[str, Any]],
                     centroid, names: List[str]) -> Dict[str, Any]:
    """Classify registered camera positions into 8 azimuth sectors + roof.

    The world frame is arbitrary, so the *front* of the vehicle cannot be
    known from geometry alone.  Default: the sector of the first registered
    view is labelled front (walk-around videos normally start at the front).
    That guess can be overridden with --front-azimuth DEGREES, or disabled
    with --front-azimuth unknown to get neutral sector labels.  Nothing is
    invented: an empty sector is reported as insufficiently observed.
    """
    np = _np()
    if not views:
        return {"sectors": {}, "messages": ["No cameras registered - no coverage to report."]}

    centers = np.array([camera_center(v["R"], v["t"]) for v in views.values()])
    directions = centers - np.asarray(centroid, dtype=np.float64).reshape(1, 3)
    azimuth = np.degrees(np.arctan2(directions[:, 1], directions[:, 0])) % 360.0
    elevation = np.degrees(np.arctan2(
        directions[:, 2],
        np.linalg.norm(directions[:, :2], axis=1) + 1e-12,
    ))
    sector_index = np.floor((azimuth + 22.5) % 360.0 / 45.0).astype(int) % 8

    counts = {name: 0 for name in SECTOR_NAMES}
    for idx in sector_index:
        counts[SECTOR_NAMES[int(idx)]] += 1
    roof_views = int((elevation > 35.0).sum())

    front_setting = getattr(cfg, "front_azimuth", "first")
    offset = 0
    labelled = True
    if isinstance(front_setting, (int, float)):
        offset = int((float(front_setting) + 22.5) // 45.0) % 8
    elif front_setting == "unknown":
        labelled = False
    else:  # "first": first registered camera defines front
        offset = int(sector_index[0])

    named = {}
    for i, name in enumerate(SECTOR_NAMES):
        # A camera at raw sector `s0` (the first registered one) must be
        # reported under "front": raw sector i is labelled (i - s0).
        ordered = SECTOR_NAMES[(i - offset) % 8]
        named[ordered] = counts[name]

    messages: List[str] = []
    if labelled:
        for direction in ("rear", "front", "left", "right"):
            if named.get(direction, 0) == 0:
                messages.append(f"{direction.capitalize()} geometry insufficiently observed.")
        if roof_views == 0:
            messages.append("Roof geometry insufficiently observed.")
    empty = [name for name, count in named.items() if count == 0]
    if empty and labelled:
        messages.append(
            "Missing view directions: " + ", ".join(sorted(empty)) +
            ". Geometry in those areas will be incomplete or interpolated - "
            "it will not be invented."
        )
    return {
        "sectors": named,
        "roof_views": roof_views,
        "front_azimuth": "labelled" if labelled else "unlabelled",
        "messages": messages,
        "camera_count": len(centers),
    }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run_sparse(cfg: ReconConfig, store: StateStore, ctx: StageContext,
               *, probe: bool = False) -> Dict[str, Any]:
    cv2, np = _cv2(), _np()
    started = time.time()
    # Reproducibility: RANSAC draws must not depend on what happened to run
    # before this call (a focal probe, a resumed job, a previous stage), or
    # two runs of identical input can register different cameras and land in
    # different minima.  The seed deliberately excludes anything focal- or
    # probe-related: probes and the final run then share one random stream,
    # so any difference between them can only come from the focal length,
    # and a run at a different assumed FOV is directly comparable.
    import zlib
    cv2.setRNGSeed(int(zlib.crc32(
        cfg.feature_fingerprint().encode()) & 0x7FFFFFFF) or 1)

    frames = selected_frames(cfg)
    if not frames:
        raise ReconstructionError(
            "Structure-from-Motion has no frames to work with.",
            suggestion="Run frame extraction and selection first.")
    # Focal probes rebuild only a contiguous block of the selection: the
    # score they produce (mean reprojection error of one coherent model)
    # is what picks the focal length, and a scattered subset is not a
    # coherent model to score.
    limit = int(getattr(cfg, "sfm_frame_limit", 0) or 0)
    if limit and len(frames) > limit:
        start = max(0, (len(frames) - limit) // 2)
        frames = frames[start:start + limit]
        ctx.note(f"focal probe over images {start + 1}..{start + len(frames)} "
                 f"of {start + limit} ({len(frames)} used)")
    names = [f.stem for f in frames]

    # --- load keypoints (descriptors are not needed after matching) -------
    keypoints: List[Any] = [None] * len(frames)
    sizes: List[Tuple[int, int]] = [(0, 0)] * len(frames)
    for i, frame in enumerate(frames):
        info = load_keypoints_only(cfg, frame)
        if info is None:
            continue
        keypoints[i] = info["kp"]
        sizes[i] = (info["width"], info["height"])

    matches: List[Tuple[int, int, Dict[str, Any]]] = []
    index_of = {name: i for i, name in enumerate(names)}
    for path in list_match_files(cfg):
        data = load_match(cfg, path)
        if data is None:
            continue
        a, b = index_of.get(data["a"]), index_of.get(data["b"])
        if a is None or b is None:
            continue
        matches.append((a, b, data))
    if not matches:
        raise ReconstructionError(
            "Structure-from-Motion failed because there are no verified image "
            "pairs to initialise from.",
            suggestion="Re-run feature matching; check that the frames overlap.")

    ctx.note(f"loaded {len(frames)} images, {len(matches)} verified pairs")

    # --- feature tracks (one 3D point per track) ----------------------------
    tracks, track_lookup = build_tracks(matches)
    usable_tracks = sum(1 for obs in tracks if len(obs) >= 2)
    if usable_tracks < 20:
        raise ReconstructionError(
            f"Structure-from-Motion failed because only {usable_tracks} "
            "multi-view feature tracks could be built.",
            suggestion="The video barely moves around the vehicle; record a "
                       "slower orbit with more overlap, or lower --frame-interval.",
            details={"tracks": len(tracks), "pairs": len(matches)},
        )

    kp_track: List[Dict[int, int]] = [dict() for _ in frames]
    for tid, obs in enumerate(tracks):
        for img, kp in obs:
            kp_track[img][kp] = tid

    views: Dict[int, Dict[str, Any]] = {}
    points_arr = np.zeros((0, 3), dtype=np.float64)
    point_track: Dict[int, int] = {}     # track id -> point row
    # Focal refined by bundle adjustment; new cameras must use it too or
    # their PnP pose would be inconsistent with the calibrated structure.
    shared_focal: Optional[float] = None

    def intrinsic_for(i: int):
        w, h = sizes[i]
        if w <= 0 or h <= 0:
            w, h = 1200, 900
        K = assumed_intrinsics(cfg, w, h)
        if shared_focal is not None and shared_focal > 0:
            K = K.copy()
            K[0, 0] = shared_focal
            K[1, 1] = shared_focal
        return K

    # --- seed pair: the strongest geometrically verified connection ---------
    # Quality alone would often pick a pair at the very end of the orbit,
    # from which the reconstruction can only grow in one direction.  Score
    # candidates by quality * reach so the initial pair also has neighbours
    # on both sides to grow into.
    neighbors: Dict[int, set] = {}
    for ma, mb, mdata in matches:
        if len(mdata["idx_a"]) < cfg.min_inliers:
            continue
        neighbors.setdefault(ma, set()).add(mb)
        neighbors.setdefault(mb, set()).add(ma)

    ranked = sorted(matches, key=lambda m: len(m[2]["idx_a"]), reverse=True)

    # Registration only ever grows inside the component the seed starts in,
    # so the seed decides *which part of the video gets reconstructed at all*.
    # Try candidates from the largest component first: seeding a smaller one
    # strands every frame outside it.  Measured on a 50 s walk-around whose
    # match graph split at a blurry section (31 / 24 / 25 frames), seeding the
    # 25-frame component registered 25/80 (31%, below the 35% gate) while the
    # 31-frame component would cover 38% of the sequence.  The quality gates
    # below are unchanged, so a weak large component still loses to a usable
    # smaller one.
    find_component, component_sizes = match_components(len(frames), matches)
    largest_root = max(component_sizes, key=component_sizes.get)
    main_pairs = [m for m in ranked if find_component(m[0]) == largest_root]
    other_pairs = [m for m in ranked if find_component(m[0]) != largest_root]
    if len(component_sizes) > 1:
        ctx.note("match graph splits into "
                 f"{len(component_sizes)} components ("
                 + ", ".join(str(s) for s in
                             sorted(component_sizes.values(), reverse=True))
                 + " images); seeding candidates from the largest first")
    seed = None
    seed_score = -1.0
    seed_quality = 0
    for a, b, data in (main_pairs + other_pairs)[:30]:
        ctx.check()
        if len(data["idx_a"]) < max(cfg.min_inliers, 16):
            # Not a break: candidates are no longer globally sorted by
            # inliers once the largest component is tried first.
            continue
        if keypoints[a] is None or keypoints[b] is None:
            continue
        K1, K2 = intrinsic_for(a), intrinsic_for(b)
        pts1 = keypoints[a][data["idx_a"], :2]
        pts2 = keypoints[b][data["idx_b"], :2]
        if len(pts1) < 12:
            continue
        try:
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K1, method=cv2.RANSAC, prob=0.999,
                threshold=cfg.ransac_threshold)
        except cv2.error:
            continue
        if E is None or mask is None or E.shape != (3, 3):
            continue
        try:
            # OpenCV 5: recoverPose(E, points1, points2, cameraMatrix, mask=...)
            _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K1, mask=mask)
        except cv2.error:
            try:
                _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K1)
            except cv2.error:
                continue
        inliers = pose_mask.ravel().astype(bool)
        if int(inliers.sum()) < cfg.min_inliers:
            continue
        view1 = {"R": np.eye(3), "t": np.zeros(3), "K": K1,
                 "rvec": np.zeros(3), "f": float(K1[0, 0])}
        view2 = {"R": R, "t": np.float64(t).ravel(), "K": K2,
                 "rvec": cv2.Rodrigues(R)[0].ravel(), "f": float(K2[0, 0])}
        X, valid = triangulate_two(view1, view2, pts1[inliers], pts2[inliers])
        if not valid.any():
            continue
        err1 = reprojection_errors(view1, X[valid], pts1[inliers][valid])
        err2 = reprojection_errors(view2, X[valid], pts2[inliers][valid])
        par = parallax_degrees(camera_center(view1["R"], view1["t"]),
                               camera_center(view2["R"], view2["t"]), X[valid])
        good = (err1 < cfg.max_reprojection_error) & \
               (err2 < cfg.max_reprojection_error) & \
               (par >= cfg.min_parallax_deg)
        quality = int(good.sum())
        reach = len((neighbors.get(a, set()) | neighbors.get(b, set()))
                    - {a, b})
        score = quality * max(1, reach)
        if quality >= 15 and score > seed_score:
            seed_score = score
            seed_quality = quality
            seed = (a, b, data, view1, view2, inliers, pts1, pts2)

    if seed is None or seed_quality < 15:
        raise ReconstructionError(
            "Structure-from-Motion failed because no initial image pair could "
            f"be calibrated (best candidate yielded {max(seed_quality, 0)} "
            "triangulated points).",
            suggestion="The frames may be too similar, too blurry, or show a "
                       "flat/degenerate scene. Record a slower, wider orbit "
                       "around the vehicle with visible texture.",
            details={"candidate_pairs": min(30, len(ranked)),
                     "best_points": max(seed_quality, 0),
                     "match_graph_components": int(len(component_sizes))},
        )

    a, b, data, view1, view2, inliers, pts1, pts2 = seed
    views[a], views[b] = view1, view2
    registered_order: List[int] = [a, b]
    # Seed points come from the same triangulation pass used everywhere else,
    # so one set of quality gates (reprojection, parallax, positive depth)
    # applies to every 3D point in the model.

    # --- observations (image, point, pixel) used by bundle adjustment --------
    obs_cam: Any = np.zeros(0, dtype=np.int64)
    obs_pt: Any = np.zeros(0, dtype=np.int64)
    obs_xy: Any = np.zeros((0, 2), dtype=np.float64)
    obs_kp: Any = np.zeros(0, dtype=np.int64)

    def refresh_observations() -> None:
        nonlocal obs_cam, obs_pt, obs_xy, obs_kp
        cams, pts, xys, kps = [], [], [], []
        for img in views:
            kp_pos = keypoints[img]
            if kp_pos is None:
                continue
            for kp, tid in kp_track[img].items():
                row = point_track.get(tid)
                if row is None or kp >= len(kp_pos):
                    continue
                cams.append(img)
                pts.append(row)
                xys.append(kp_pos[kp, :2])
                kps.append(kp)
        obs_cam = np.array(cams, dtype=np.int64)
        obs_pt = np.array(pts, dtype=np.int64)
        obs_xy = np.array(xys, dtype=np.float64).reshape(-1, 2)
        obs_kp = np.array(kps, dtype=np.int64)

    def append_points(block: List[np.ndarray]) -> None:
        nonlocal points_arr
        if not block:
            return
        points_arr = np.vstack([points_arr,
                                np.array(block, dtype=np.float64).reshape(-1, 3)])

    def triangulate_missing() -> int:
        """Triangulate every track observed by >= 2 registered cameras.

        One pass covers the seed pair and every later registration, and it
        picks the view pair with the largest baseline so the triangulation
        angle is as good as the data allows (sections 11 and 14).

        Every other observation of the same track is then checked against
        the resulting 3D point.  A track is the union of independent
        pairwise matches, so a single unlucky RANSAC on a nearly
        non-overlapping pair can graft a wrong pixel onto an otherwise good
        track; those observations are dropped here, which is what keeps
        them from poisoning bundle adjustment and PnP later.
        """
        block: List[np.ndarray] = []
        pruned = 0
        for tid, obs in enumerate(tracks):
            if tid in point_track:
                continue
            usable = [(img, kp) for img, kp in obs
                      if img in views and keypoints[img] is not None
                      and kp < len(keypoints[img])
                      and kp_track[img].get(kp) == tid]
            if len(usable) < 2:
                continue
            centers = {img: camera_center(views[img]["R"], views[img]["t"])
                       for img, _ in usable}
            best_pair, best_baseline = None, -1.0
            for x in range(len(usable)):
                for y in range(x + 1, len(usable)):
                    i1, _ = usable[x]
                    i2, _ = usable[y]
                    base = float(np.linalg.norm(centers[i1] - centers[i2]))
                    if base > best_baseline:
                        best_pair, best_baseline = (x, y), base
            if best_pair is None:
                continue
            (i1, k1) = usable[best_pair[0]]
            (i2, k2) = usable[best_pair[1]]
            p1 = keypoints[i1][k1, :2].reshape(1, 2)
            p2 = keypoints[i2][k2, :2].reshape(1, 2)
            X1, valid = triangulate_two(views[i1], views[i2], p1, p2)
            if not valid[0]:
                continue
            X = X1[0]
            if not np.isfinite(X).all():
                continue
            if not (_positive_depth(views[i1], X) and _positive_depth(views[i2], X)):
                continue
            e1 = float(reprojection_errors(views[i1], X.reshape(1, 3), p1)[0])
            e2 = float(reprojection_errors(views[i2], X.reshape(1, 3), p2)[0])
            par = float(parallax_degrees(centers[i1], centers[i2],
                                         X.reshape(1, 3))[0])
            if e1 > cfg.max_reprojection_error or \
                    e2 > cfg.max_reprojection_error or \
                    par < cfg.min_parallax_deg:
                continue

            # --- multi-view consistency check -------------------------------
            keep: List[Tuple[int, int]] = []
            for img, kp in usable:
                e = float(reprojection_errors(
                    views[img], X.reshape(1, 3),
                    keypoints[img][kp, :2].reshape(1, 2))[0])
                if e <= cfg.track_max_reprojection_error and \
                        _positive_depth(views[img], X):
                    keep.append((img, kp))
                else:
                    kp_track[img].pop(kp, None)   # drop the bad pixel
                    pruned += 1
            if len(keep) < 2:
                continue
            point_track[tid] = len(points_arr) + len(block)
            block.append(X)
        append_points(block)
        if pruned:
            ctx.note(f"rejected {pruned} inconsistent track observations")
        return len(block)

    seed_added = triangulate_missing()
    refresh_observations()
    ctx.note(f"seed pair {names[a]} + {names[b]}: {seed_added} points")
    rejected: List[Dict[str, Any]] = []
    # Declared before the loop (and re-cleared each pass) so the report can
    # always quote a refusal reason, even if the very first pass found no
    # candidates at all.
    failures: Dict[int, str] = {}
    calibrated = False

    # --- incremental registration -------------------------------------------
    while True:
        ctx.check()
        candidates: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for img in range(len(frames)):
            if img in views or keypoints[img] is None:
                continue
            obj_pts, img_pts = [], []
            for kp, tid in kp_track[img].items():
                row = point_track.get(tid)
                if row is None or kp >= len(keypoints[img]):
                    continue
                obj_pts.append(points_arr[row])
                img_pts.append(keypoints[img][kp, :2])
            if len(obj_pts) >= cfg.min_pnp_correspondences:
                candidates.append((img, np.float64(obj_pts), np.float64(img_pts)))
        if not candidates:
            # Say why the frontier stalled instead of stopping silently: a
            # track only yields a 2D-3D pair once two of its images are
            # registered, so a candidate that is linked to only one camera
            # can never be positioned (section 15).
            for img in range(len(frames)):
                if img in views or keypoints[img] is None:
                    continue
                linked = with_points = 0
                for kp, tid in kp_track[img].items():
                    linked += 1
                    if point_track.get(tid) is not None:
                        with_points += 1
                if linked:
                    ctx.note(f"{names[img]}: {with_points}/{linked} matched "
                             f"features belong to a triangulated track "
                             f"(needs >= {cfg.min_pnp_correspondences})")
                else:
                    ctx.note(f"{names[img]}: no verified match to a "
                             f"registered camera")
            break
        candidates.sort(key=lambda c: -len(c[1]))

        failures: Dict[int, str] = {}
        registered_any = False
        for img, obj_pts, img_pts in candidates[:8]:
            ctx.check()
            K = intrinsic_for(img)
            try:
                ok_pnp, rvec, tvec, inl = cv2.solvePnPRansac(
                    obj_pts.reshape(-1, 1, 3), img_pts.reshape(-1, 1, 2),
                    K, None, iterationsCount=150, reprojectionError=3.0,
                    confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
                )
            except cv2.error:
                failures[img] = "solvePnPRansac failed (degenerate geometry)"
                continue
            if not ok_pnp or inl is None:
                _dump_pnp(cfg, names[img], obj_pts, img_pts, K)
                failures[img] = (
                    f"solvePnPRansac found no consensus pose across "
                    f"{len(obj_pts)} correspondences"
                    f"{_pnp_diagnostic(obj_pts, img_pts, K)}")
                continue
            # solvePnPRansac returns inliers as an (N,1) array; flatten it
            # before using it as an index or the shapes silently broadcast.
            inl = np.asarray(inl).ravel().astype(np.int64)
            got = len(inl)
            if got < cfg.min_pnp_inliers or \
                    got < cfg.min_pnp_inlier_ratio * len(obj_pts):
                failures[img] = (f"PnP produced {got} inliers out of "
                                 f"{len(obj_pts)} correspondences "
                                 f"(needs >= {cfg.min_pnp_inliers} and >= "
                                 f"{cfg.min_pnp_inlier_ratio:.0%})")
                continue
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj_pts[inl].reshape(-1, 1, 3),
                    img_pts[inl].reshape(-1, 1, 2), K, None, rvec, tvec)
            except cv2.error:
                pass
            R, _ = cv2.Rodrigues(rvec)
            view = {"R": np.float64(R), "t": np.float64(tvec).ravel(),
                    "K": K, "rvec": np.float64(rvec).ravel(),
                    "f": float(K[0, 0])}
            # Judge the pose on its RANSAC *inliers* only: outlier matches
            # are expected in the raw correspondence set and must not be
            # counted against an otherwise good pose.
            errs = reprojection_errors(view, obj_pts[inl], img_pts[inl])
            if float(errs.mean()) > cfg.max_reprojection_error:
                failures[img] = (f"mean reprojection error of {len(inl)} "
                                 f"PnP inliers {errs.mean():.1f} px exceeds "
                                 f"{cfg.max_reprojection_error} px")
                continue

            # --- pose sanity against the working scale ---------------------
            # PnP can "explain" a few nearly collinear points with the
            # camera at a nonsense distance.  Such a camera survives bundle
            # adjustment because the robust loss down-weights exactly the
            # residuals that would expose it, so it has to be refused here.
            # Two independent references keep this scale free: how far the
            # registered cameras sit from the structure, and how big the
            # structure itself is.
            if views and len(point_track):
                rows = np.fromiter(point_track.values(), dtype=np.int64,
                                   count=len(point_track))
                structure = points_arr[rows]
                centroid = structure.mean(axis=0)
                object_radius = float(np.median(
                    np.linalg.norm(structure - centroid, axis=1)))
                known = np.array([camera_center(v["R"], v["t"])
                                  for v in views.values()])
                typical = float(np.median(
                    np.linalg.norm(known - centroid, axis=1)))
                distance = float(np.linalg.norm(
                    camera_center(view["R"], view["t"]) - centroid))
                too_far = typical > 0 and \
                    distance > cfg.camera_radius_ratio_max * typical
                too_close = object_radius > 0 and \
                    distance < cfg.camera_radius_ratio_min * object_radius
                if too_far or too_close:
                    failures[img] = (
                        f"PnP pose puts the camera {distance:.2f} units from "
                        f"the structure (registered cameras ~{typical:.2f}, "
                        f"structure radius {object_radius:.2f}) - rejected "
                        f"as degenerate")
                    continue

            views[img] = view
            registered_order.append(img)
            registered_any = True
            ctx.note(f"registered {names[img]} ({len(inl)} PnP inliers, "
                     f"{len(views)}/{len(frames)} cameras)")

            # Triangulate every track now seen by at least two registered
            # cameras - not just the ones touching the new camera - so the
            # 2D-3D pool keeps growing for the next registration.
            added = triangulate_missing()
            ctx.note(f"triangulated {added} new points (total "
                     f"{len(points_arr)})")
            refresh_observations()

            # Calibration pass: with several views the focal length finally
            # becomes identifiable, so scan it once before PnP continues
            # (guessing FOV is only an initial estimate).
            if not calibrated and len(views) >= 4:
                calibrated = True
                calib_points = sorted(point_track.values())
                if len(calib_points) > cfg.ba_max_points:
                    rng = np.random.default_rng(5)
                    calib_points = sorted(int(x) for x in rng.choice(
                        calib_points, size=cfg.ba_max_points, replace=False))
                calib = calibrate_focal(
                    views, points_arr, obs_cam, obs_pt, obs_xy,
                    camera_ids=list(views), point_ids=calib_points,
                    anchor_id=registered_order[0], cfg=cfg,
                )
                ctx.note(f"calibration: {calib}")
                if cfg.ba_refine_intrinsics and calib.get("performed"):
                    shared_focal = float(views[registered_order[0]]["f"])

            # Local bundle adjustment over a sliding window of cameras.
            # kp_track[c] maps *keypoint index -> track id*, so the track ids
            # are the values: reading the keys silently passed keypoint
            # indices as point ids, which left the window with a handful of
            # coincidental matches and made every local BA a no-op.
            window = registered_order[-6:]
            window_tids = set()
            for c in window:
                window_tids.update(kp_track[c].values())
            local_points = sorted(point_track[t] for t in window_tids
                                  if t in point_track)
            if len(local_points) > cfg.ba_max_points:
                rng = np.random.default_rng(3)
                local_points = sorted(int(x) for x in rng.choice(
                    local_points, size=cfg.ba_max_points, replace=False))
            ba = bundle_adjust(
                views, points_arr, obs_cam, obs_pt, obs_xy,
                camera_ids=window, point_ids=local_points,
                anchor_id=registered_order[0],
                refine_f=False,
                fallback_f=float(view["f"]),
                max_nfev=cfg.ba_max_nfev,
                # Keep every observation of the window's points, including
                # the ones from cameras outside the window: otherwise the
                # window refits shared structure to itself and the global
                # model drifts apart while local BA happily reports success.
                all_observers=True,
            )
            ctx.note(f"local BA after {names[img]}: {ba}")
            break  # one successful registration per outer iteration

        if not registered_any:
            # Persist *why* the best candidates were refused (section 25).
            for img, obj_pts, _ in candidates[:8]:
                rejected.append({
                    "image": names[img],
                    "reason": failures.get(
                        img,
                        f"not enough 2D-3D correspondences ({len(obj_pts)} "
                        f"available, needs >= {cfg.min_pnp_correspondences})"),
                })
            break

    unregistered = [names[i] for i in range(len(frames)) if i not in views]
    ctx.note(f"registration finished: {len(views)}/{len(frames)} cameras, "
             f"{len(points_arr)} points")

    if len(views) < 2 or len(points_arr) < 30:
        raise ReconstructionError(
            f"Structure-from-Motion failed because only {len(views)} of "
            f"{len(frames)} images could be registered and "
            f"{len(points_arr)} points were triangulated.",
            suggestion="Check sfm/sfm.json for per-camera reasons; the video "
                       "probably lacks overlap between views. Re-record a "
                       "slower, continuous orbit around the vehicle.",
            details={"registered": len(views), "images": len(frames),
                     "points": int(len(points_arr)),
                     "unregistered_reasons": rejected[:10]},
        )

    # --- final bundle adjustment --------------------------------------------
    final_points = list(range(len(points_arr)))
    if len(final_points) > cfg.ba_max_points:
        rng = np.random.default_rng(11)
        final_points = sorted(int(x) for x in rng.choice(
            final_points, size=cfg.ba_max_points, replace=False))
    final_ba = bundle_adjust(
        views, points_arr, obs_cam, obs_pt, obs_xy,
        camera_ids=list(views.keys()),
        point_ids=final_points,
        anchor_id=registered_order[0],
        refine_f=bool(cfg.ba_refine_intrinsics),
        fallback_f=float(views[registered_order[0]]["f"]),
        max_nfev=max(cfg.ba_max_nfev, 80),
        f_prior_rel=cfg.ba_focal_prior,
    )
    ctx.note(f"final BA: {final_ba}")

    # The final model sees every camera and observation, so it is the best
    # place to settle the focal length: one more continuation walk, then the
    # camera quality pass below is measured against that solution.
    if cfg.ba_refine_intrinsics and len(views) >= 4:
        final_calib = calibrate_focal(
            views, points_arr, obs_cam, obs_pt, obs_xy,
            camera_ids=list(views.keys()), point_ids=final_points,
            anchor_id=registered_order[0], cfg=cfg,
        )
        ctx.note(f"final focal calibration: {final_calib}")
        if final_calib.get("performed"):
            final_ba = final_calib.get("refine", final_ba)
            refresh_observations()

    # --- iterative observation filtering ------------------------------------
    # Bundle adjustment's robust loss down-weights an observation it cannot
    # explain but never removes it, so a wrong match keeps tugging on its
    # camera and its neighbours: the model stays self-consistent (a couple of
    # pixels) while the trajectory silently deforms - measured as 0.5 m of
    # error in the middle of the orbit but 11 m at both ends.  Dropping the
    # observations that are still bad after a refit, and the points that lose
    # a second view, is what lets the remaining geometry pull straight.
    filter_log: List[Dict[str, Any]] = []
    for round_index in range(max(0, int(cfg.ba_filter_rounds))):
        refresh_observations()
        total_obs = int(len(obs_cam))
        if total_obs == 0:
            break
        errors = np.empty(total_obs, dtype=np.float64)
        for c in views:
            mask = obs_cam == c
            if mask.any():
                errors[mask] = reprojection_errors(
                    views[c], points_arr[obs_pt[mask]], obs_xy[mask])
        bad = np.nonzero(errors > cfg.track_max_reprojection_error)[0]
        record: Dict[str, Any] = {
            "round": round_index + 1,
            "observations": total_obs,
            "median_error_px": round(float(np.median(errors)), 3),
            "worst_error_px": round(float(errors.max()), 2),
            "removed": int(len(bad)),
        }
        filter_log.append(record)
        if len(bad) == 0:
            break
        for idx in bad:
            kp_track[int(obs_cam[idx])].pop(int(obs_kp[idx]), None)
        refresh_observations()

        # A point left with a single view defines no geometry: forget it so
        # it neither enters the refit nor the exported cloud.  Rows in
        # points_arr are never renumbered - obs_pt keeps addressing them.
        _, counts = np.unique(obs_pt, return_counts=True)
        dead_rows = set(int(r) for r in np.unique(obs_pt)[counts < 2])
        if dead_rows:
            for tid in [t for t, r in point_track.items() if r in dead_rows]:
                del point_track[tid]
            refresh_observations()

        ba = bundle_adjust(
            views, points_arr, obs_cam, obs_pt, obs_xy,
            camera_ids=list(views.keys()), point_ids=final_points,
            anchor_id=registered_order[0],
            refine_f=bool(cfg.ba_refine_intrinsics),
            fallback_f=float(views[registered_order[0]]["f"]),
            max_nfev=cfg.ba_max_nfev,
            f_prior_rel=cfg.ba_focal_prior,
        )
        record["refit"] = bool(ba.get("performed"))
        record["mean_reprojection_after_px"] = ba.get(
            "mean_reprojection_after_px")
        ctx.note(
            f"observation filter round {record['round']}: removed "
            f"{record['removed']} of {total_obs} observations above "
            f"{cfg.track_max_reprojection_error}px "
            f"(median was {record['median_error_px']} px); refit -> "
            f"{record['mean_reprojection_after_px']} px")
        if not ba.get("performed"):
            break
        if len(bad) < max(10, int(0.01 * total_obs)):
            break
    if filter_log:
        ctx.note(f"observation filtering: {filter_log}")

    # --- per-camera quality control (section 12) ------------------------------
    # A degenerate PnP pose can survive registration: the robust loss inside
    # bundle adjustment down-weights exactly the residuals that would expose
    # it, so one camera can end up thousands of units away with the rest of
    # the model barely affected.  Measure every camera, drop the ones that do
    # not fit, and let the model settle again.
    def camera_error_stats() -> Dict[int, Tuple[float, float, int, np.ndarray]]:
        stats: Dict[int, Tuple[float, float, int, np.ndarray]] = {}
        for c in sorted(views):
            mask = obs_cam == c
            if not mask.any():
                continue
            err = reprojection_errors(
                views[c], points_arr[obs_pt[mask]], obs_xy[mask])
            stats[c] = (float(err.mean()), float(np.median(err)),
                        int(mask.sum()), err)
        return stats

    refresh_observations()
    camera_stats = camera_error_stats()
    dropped: List[Dict[str, Any]] = []
    if len(views) > 3:
        for c in sorted(camera_stats):
            mean_c, _median_c, n_obs, _err = camera_stats[c]
            if (n_obs >= cfg.camera_min_observations
                    and mean_c > cfg.camera_max_reprojection_error):
                dropped.append({
                    "image": names[c],
                    "mean_reprojection_error_px": round(mean_c, 2),
                    "observations": n_obs,
                    "reason": (f"dropped after bundle adjustment: mean "
                               f"reprojection error {mean_c:.1f} px over "
                               f"{n_obs} observations "
                               f"(limit {cfg.camera_max_reprojection_error} px)"),
                })
                del views[c]
                if c in registered_order:
                    registered_order.remove(c)
        if dropped:
            for entry in dropped:
                ctx.note(f"dropping {entry['image']}: "
                         f"{entry['reason']}")
                rejected.append({"image": entry["image"],
                                 "reason": entry["reason"]})
            refresh_observations()
            camera_stats = camera_error_stats()
            if len(views) >= 2 and camera_stats:
                final_ba = bundle_adjust(
                    views, points_arr, obs_cam, obs_pt, obs_xy,
                    camera_ids=list(views.keys()),
                    point_ids=final_points,
                    anchor_id=registered_order[0],
                    refine_f=bool(cfg.ba_refine_intrinsics),
                    fallback_f=float(views[registered_order[0]]["f"]),
                    max_nfev=max(cfg.ba_max_nfev, 80),
                    f_prior_rel=cfg.ba_focal_prior,
                )
                ctx.note(f"final BA after dropping {len(dropped)} cameras: "
                         f"{final_ba}")
                refresh_observations()
                camera_stats = camera_error_stats()

    # --- reprojection statistics (section 12) ---------------------------------
    error_chunks = [entry[3] for entry in camera_stats.values()]
    all_errors = np.concatenate(error_chunks) if error_chunks else np.zeros(0)
    mean_err = float(all_errors.mean()) if len(all_errors) else float("nan")
    median_err = float(np.median(all_errors)) if len(all_errors) else float("nan")

    # Points left without any observation after the repair are not backed by
    # evidence any more, so they are kept out of the exported cloud.
    observed_points = np.zeros(len(points_arr), dtype=bool)
    if len(obs_cam):
        observed_points[np.unique(obs_pt)] = True
    export_points = points_arr[observed_points]
    unregistered = [names[i] for i in range(len(frames)) if i not in views]

    # --- connected component analysis (sections 10 and 12) -------------------
    # Same helper the seed used, so the report and the seed decision always
    # describe the same graph.
    find, comp_sizes = match_components(len(frames), matches)
    main_root = max(comp_sizes, key=comp_sizes.get)
    main_size = comp_sizes[main_root]
    registered_in_main = sum(1 for img in views if find(img) == main_root)

    # --- validation report (section 12) ---------------------------------------
    registered = len(views)
    total = len(frames)
    # Cameras repaired away must still be visible in the report: they are
    # exactly the images a human needs to re-record.
    dropped_names = {entry["image"] for entry in dropped}
    reasons = list(dropped) + [r for r in rejected
                               if r.get("image") not in dropped_names]
    # Every unregistered image needs a reason, not just the handful the final
    # registration pass happened to score - the failure message sends readers
    # to this list, and returning it empty sends them nowhere.  Membership of
    # a match-graph component with no registered camera is the diagnosis that
    # actually explains a stalled reconstruction: no amount of PnP tuning can
    # join such an image, because verified pairs never reach it.
    covered = {entry.get("image") for entry in reasons}
    for i in range(len(frames)):
        if i in views or names[i] in covered:
            continue
        root = find(i)
        registered_in_comp = sum(1 for img in views if find(img) == root)
        if registered_in_comp == 0:
            reason = (
                f"its match-graph component ({comp_sizes[root]} images) "
                f"holds no registered camera, so no verified pair bridges it "
                f"to the reconstruction - usually motion blur or a fast "
                f"camera move across that section")
        else:
            reason = failures.get(
                i, "no 2D-3D correspondences with the registered cameras "
                   "(its matches to them do not form triangulated tracks)")
        reasons.append({"image": names[i], "reason": reason})

    coverage = analyze_coverage(
        cfg, views,
        np.median(export_points, axis=0) if len(export_points)
        else np.zeros(3),
        names)
    report: Dict[str, Any] = {
        "registered_images": registered,
        "total_images": total,
        "registered_percent": round(100.0 * registered / max(1, total), 1),
        "sparse_points": int(len(export_points)),
        "observations": int(len(obs_cam)),
        "average_reprojection_error_px": round(mean_err, 3),
        "median_reprojection_error_px": round(median_err, 3),
        "cameras": [
            {"name": names[c],
             "mean_reprojection_error_px": round(camera_stats[c][0], 3),
             "median_reprojection_error_px": round(camera_stats[c][1], 3),
             "observations": camera_stats[c][2]}
            for c in sorted(camera_stats)
        ],
        "dropped_cameras": dropped,
        "largest_component_images": int(main_size),
        "main_component_registered_images": int(registered_in_main),
        "camera_graph_components": int(len(comp_sizes)),
        "bundle_adjustment": final_ba,
        "observation_filtering": filter_log,
        "unregistered_images": unregistered,
        "unregistered_reasons": reasons,
        "coverage": coverage,
        "elapsed_seconds": round(time.time() - started, 1),
        "cpu_only": True,
    }
    report["summary"] = (
        f"Registered images: {registered} / {total} | "
        f"Sparse points: {len(export_points)} | "
        f"Average reprojection error: {mean_err:.2f} px | "
        f"Match graph: {len(comp_sizes)} "
        f"component{'s' if len(comp_sizes) != 1 else ''}, largest "
        f"{main_size} images with {registered_in_main} registered")
    if main_size < total:
        sizes_desc = ", ".join(str(s) for s in
                               sorted(comp_sizes.values(), reverse=True))
        report["disconnected_graph_notice"] = (
            f"The match graph splits into {len(comp_sizes)} disconnected "
            f"components ({sizes_desc} images). Registration can only grow "
            f"inside the component it starts in, so the {total - main_size} "
            f"images outside the largest one cannot join without more "
            f"verified pairs bridging those sections.")

    # --- exports: PLY clouds, PNG preview, sfm.json, report -------------------
    # A focal probe writes nothing at all, so a probe run can never be
    # mistaken for the real reconstruction (no stale cloud, no sfm.json).
    exports: List[Path] = []
    if not probe:
        exports = export_sparse(cfg, ctx, frames, views, export_points,
                                report, registered_order[0])

        cfg.d("sfm").mkdir(parents=True, exist_ok=True)
        sfm_payload = {
            "images": [
                {
                    "name": names[i],
                    "index": i,
                    "width": sizes[i][0],
                    "height": sizes[i][1],
                    "registered": True,
                    "R": [round(float(v), 9) for v in views[i]["R"].ravel()],
                    "t": [round(float(v), 9) for v in views[i]["t"].ravel()],
                    "center": [round(float(v), 9) for v in
                               camera_center(views[i]["R"], views[i]["t"])],
                    "focal": float(views[i]["f"]),
                }
                for i in sorted(views)
            ],
            "unregistered": unregistered,
            "unregistered_reasons": reasons,
            "points": int(len(export_points)),
            "observations": int(len(obs_cam)),
            "report": report,
        }
        cfg.d("sfm", "sfm.json").write_text(
            json.dumps(sfm_payload, indent=2), encoding="utf-8")
        exports.append(cfg.d("sfm", "sfm.json"))

        # Section 6: never let a missing viewpoint pass silently.
        for message in coverage.get("messages", []):
            ctx.note(message)
            ctx.log(message if message.isupper() else message,
                    stage="COVERAGE", operation="analyze", status="warning"
                    if "insufficiently" in message else "info")

    # Section 12: reject the reconstruction when too little registered.
    ratio = registered / max(1, total)
    if registered < cfg.min_registered_images or ratio < cfg.min_registered_ratio:
        # Say what actually went wrong.  "Not enough overlap" is the guess a
        # disconnected graph gets labelled with, but the component structure
        # is a fact: when verified pairs split the sequence, frames in a
        # component with no registered camera can never join, and the useful
        # advice (bridge the weak section) is completely different from
        # "re-record a wider orbit".
        component_sizes = sorted(comp_sizes.values(), reverse=True)
        if len(component_sizes) > 1:
            suggestion = (
                f"The match graph splits into {len(component_sizes)} "
                f"disconnected components ({', '.join(map(str, component_sizes))}"
                f" images). Frames in a component with no registered camera "
                "cannot join - the pairs that would bridge that section failed "
                "geometric verification, usually because of motion blur or a "
                "fast camera move there. Re-record that section slowly and "
                "steadily, or extract finer frames (--quality medium) so "
                "neighbours overlap across it; see sfm/sfm.json "
                "unregistered_reasons for the per-image diagnosis.")
        else:
            suggestion = (
                "Read sfm/sfm.json unregistered_reasons for the per-camera "
                "diagnosis; usually the video does not orbit the vehicle with "
                "enough overlap. Re-record a slower, wider walk-around.")
        raise ReconstructionError(
            f"Structure-from-Motion failed because only {registered} of "
            f"{total} images could be registered "
            f"({ratio * 100:.0f}% below the required "
            f"{cfg.min_registered_ratio * 100:.0f}%).",
            suggestion=suggestion,
            details=report,
        )

    if probe:
        report["probe"] = True
        return report
    report["exports"] = [str(p.relative_to(cfg.project_dir)) for p in exports]
    ctx.result(report)
    return report


def run_sparse_multistart(cfg: ReconConfig, store: StateStore,
                          ctx: StageContext) -> Dict[str, Any]:
    """Pick the focal length by rebuilding a small model several times.

    The FOV prior for video without EXIF is a guess, and a wrong guess is
    not something later stages can repair: bundle adjustment keeps it (the
    points absorb the error - that direction is flat) and a profile walk
    cannot leave the basin the model was built in.  Measured on the
    synthetic test set, a focal length that was 10% too long still fitted
    the images at 1.1 px median reprojection while putting camera positions
    48% away from ground truth.  The only reliable test is to rebuild.

    So a contiguous sub-model is rebuilt at each candidate focal length
    (focal probes - cheap, and they write nothing) and scored by how well it
    explains its own observations; the winner is then used for the real
    reconstruction.  If the best candidate sits on the edge of the grid the
    grid is extended outwards, because the true value may lie beyond it.
    """
    scales = list(cfg.focal_probe_scales or ())
    if not cfg.focal_multistart or cfg.focal_override_px > 0 or len(scales) < 2:
        return run_sparse(cfg, store, ctx)

    frames = selected_frames(cfg)
    if len(frames) <= max(4, cfg.focal_probe_frames):
        return run_sparse(cfg, store, ctx)

    probe_cfg0 = dataclasses.replace(cfg, sfm_frame_limit=cfg.focal_probe_frames)
    info = load_keypoints_only(cfg, frames[len(frames) // 2])
    w, h = (int(info["width"]), int(info["height"])) if info else (1200, 900)
    base_f = float(assumed_intrinsics(probe_cfg0, w, h)[0, 0])

    # A probe that registers only a few of its images is not comparable to
    # one that registers almost all of them: a shorter model fits its own
    # observations *better*, so scoring on reprojection alone would happily
    # pick a focal length that threw images away.
    required = max(cfg.min_registered_images,
                   int(math.ceil(0.75 * cfg.focal_probe_frames)))
    results: Dict[float, Dict[str, Any]] = {}

    def probe(scale: float) -> None:
        focal = base_f * float(scale)
        probe_cfg = dataclasses.replace(cfg, focal_override_px=focal,
                                         sfm_frame_limit=cfg.focal_probe_frames)
        entry: Dict[str, Any] = {"scale": round(float(scale), 4),
                                 "focal_px": round(focal, 1)}
        try:
            rep = run_sparse(probe_cfg, store, ctx, probe=True)
        except ReconstructionError as exc:
            entry["usable"] = False
            entry["reason"] = str(exc).splitlines()[0][:160]
        else:
            registered = int(rep.get("registered_images", 0))
            # Hoisted out of entry.update(): the dict literal below is fully
            # evaluated before update() runs, so entry['total'] would raise
            # KeyError the moment an under-registering probe builds its
            # reason - exactly the case this message exists for.
            total = int(rep.get("total_images", cfg.focal_probe_frames))
            entry.update({
                "usable": registered >= required,
                "registered": registered,
                "total": total,
                "mean_reprojection_px": rep.get(
                    "average_reprojection_error_px"),
                "median_reprojection_px": rep.get(
                    "median_reprojection_error_px"),
                "sparse_points": rep.get("sparse_points"),
                "reason": (None if registered >= required else
                           f"only {registered} of {total} images "
                           f"registered (need {required})"),
            })
        results[float(scale)] = entry
        if not entry.get("usable"):
            detail = f"unusable - {entry.get('reason')}"
        else:
            repro = entry.get("mean_reprojection_px")
            detail = (f"{repro:.2f} px over "
                      f"{entry.get('registered')}/{entry.get('total')} images"
                      if isinstance(repro, (int, float)) else
                      f"reprojection n/a over "
                      f"{entry.get('registered')}/{entry.get('total')} images")
        ctx.note(f"focal probe f={focal:.1f} px ({scale:+.0%}): " + detail)

    for scale in scales:
        probe(scale)

    # Extend outwards while the winner sits on the edge of the grid.
    for _ in range(2):
        good = {s: e for s, e in results.items() if e.get("usable")}
        if not good:
            break
        ordered = sorted(good)
        best = min(good, key=lambda s: (good[s]["mean_reprojection_px"],
                                        -good[s]["registered"], abs(s - 1.0)))
        if best == ordered[0]:
            nxt = best * 0.85
        elif best == ordered[-1]:
            nxt = best * 1.18
        else:
            break
        if nxt in results:
            break
        probe(nxt)

    good = {s: e for s, e in results.items() if e.get("usable")}
    chosen = min(good, key=lambda s: (good[s]["mean_reprojection_px"],
                                      -good[s]["registered"], abs(s - 1.0))) \
        if good else None
    verdict = (f"focal multi-start: chose {base_f * chosen:.1f} px "
               f"({(chosen - 1) * 100:+.1f}% vs the prior)"
               if chosen is not None else
               "focal multi-start: no usable probe; keeping the "
               f"{cfg.initial_fov_degrees:.0f} deg FOV prior")
    ctx.note(verdict)
    ctx.log(verdict, stage="SFM", operation="calibrate_focal", status="info")

    final_cfg = cfg if chosen is None else dataclasses.replace(
        cfg, focal_override_px=base_f * chosen)
    report = run_sparse(final_cfg, store, ctx)
    report["focal_multistart"] = {
        "base_focal_px": round(base_f, 1),
        "chosen_scale": chosen,
        "chosen_focal_px": (round(base_f * chosen, 1)
                            if chosen is not None else None),
        "required_registered": required,
        "probes": [results[s] for s in sorted(results)],
    }
    # reconstruction_report.json is written inside run_sparse, i.e. *before*
    # these scores exist, so the one number a human wants when a focal guess
    # looks wrong - what every probe scored and why this one won - was the
    # single field missing from the report a reader is pointed at.  Merge it
    # into the file that is already on disk; a missing report must never fail
    # a reconstruction that has otherwise finished.
    try:
        on_disk = json.loads(cfg.report_path.read_text(encoding="utf-8"))
        on_disk["focal_multistart"] = report["focal_multistart"]
        cfg.report_path.write_text(json.dumps(on_disk, indent=2),
                                   encoding="utf-8")
    except (OSError, ValueError) as exc:
        ctx.note(f"could not attach focal multi-start scores to "
                 f"{cfg.report_path.name}: {exc}")
    return report


# ---------------------------------------------------------------------------
# Exports: PLY point cloud, camera centres, PNG preview, report
# ---------------------------------------------------------------------------
def export_sparse(cfg: ReconConfig, ctx: StageContext, frames, views,
                  points, report, anchor: int) -> List[Path]:
    np, cv2 = _np(), _cv2()
    from PIL import Image, ImageDraw

    sparse_dir = cfg.d("sparse")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    # --- colours sampled from the first camera that sees each point ---------
    colors = np.full((len(points), 3), 200, dtype=np.uint8)
    remaining = np.ones(len(points), dtype=bool)
    for img in sorted(views):
        if not remaining.any() or len(points) == 0:
            break
        view = views[img]
        image = cv2.imread(str(frames[img]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        projected = project_points(points[remaining],
                                   view["rvec"], view["t"], view["K"])
        projected = np.asarray(projected)
        inside = ((projected[:, 0] >= 0) & (projected[:, 0] < w) &
                  (projected[:, 1] >= 0) & (projected[:, 1] < h))
        src = np.nonzero(remaining)[0][inside]
        if len(src):
            xs = projected[inside, 0].astype(np.int32)
            ys = projected[inside, 1].astype(np.int32)
            bgr = image[ys, xs]
            colors[src, 0] = bgr[:, 2]
            colors[src, 1] = bgr[:, 1]
            colors[src, 2] = bgr[:, 0]
            remaining[src] = False
        del image

    ply_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])

    def write_ply(path: Path, xyz, rgb) -> None:
        vertex = np.empty(len(xyz), dtype=ply_dtype)
        vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(xyz)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        with path.open("wb") as handle:
            handle.write(header.encode("ascii"))
            vertex.tofile(handle)

    if len(points):
        write_ply(sparse_dir / "sparse_point_cloud.ply", points, colors)
        written.append(sparse_dir / "sparse_point_cloud.ply")

    # --- camera centres (green; anchor camera red) --------------------------
    ids = sorted(views)
    centers = np.array([camera_center(views[i]["R"], views[i]["t"]) for i in ids])
    cam_colors = np.zeros((len(ids), 3), dtype=np.uint8)
    cam_colors[:, 1] = 230
    cam_colors[:, 0] = 60
    for slot, i in enumerate(ids):
        if i == anchor:
            cam_colors[slot] = np.array([255, 60, 60], dtype=np.uint8)
    if len(ids):
        write_ply(sparse_dir / "cameras.ply", centers, cam_colors)
        written.append(sparse_dir / "cameras.ply")

    # --- PNG preview: sparse points drawn over the anchor frame -------------
    try:
        if len(points) and anchor in views:
            view = views[anchor]
            base = cv2.imread(str(frames[anchor]), cv2.IMREAD_COLOR)
            if base is not None:
                projected = project_points(points, view["rvec"], view["t"],
                                           view["K"])
                projected = np.asarray(projected)
                h, w = base.shape[:2]
                inside = ((projected[:, 0] >= 0) & (projected[:, 0] < w) &
                          (projected[:, 1] >= 0) & (projected[:, 1] < h))
                overlay = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(overlay)
                for x, y, c in zip(projected[inside, 0],
                                   projected[inside, 1], colors[inside]):
                    draw.point((int(x), int(y)),
                               fill=(int(c[0]), int(c[1]), int(c[2])))
                preview = sparse_dir / "sparse_preview.png"
                overlay.save(preview)
                written.append(preview)
    except Exception:
        # A preview is a convenience; never fail a run over it.
        pass

    # --- reconstruction_report.json (section 12) -----------------------------
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_only": True,
        "summary": report.get("summary"),
        **{k: v for k, v in report.items() if k != "summary"},
        "exports": [str(p.relative_to(cfg.project_dir)) for p in written],
    }
    cfg.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    written.append(cfg.report_path)
    ctx.note(f"exported {len(points)} points and {len(views)} cameras")
    return written
