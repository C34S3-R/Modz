"""Phase 7: UV atlas + texture bake (frames -> texture.png).

The low-poly mesh carries only averaged vertex colours; a BUSSID mod
wants a real texture.  xatlas is unavailable on this host, so the atlas
is hand-rolled in four deterministic steps:

1. **Dominant view per triangle.**  For every frame the triangle must be
   front-facing, fully inside the image, and large enough to matter;
   where a depth map exists it must also be *depth-consistent* with that
   frame inside the same fusion band the mesh used, so occluded or
   contradictory pixels never get painted.  The largest passing
   projection wins, and every face records the views that come within
   80 % of its own best (its *candidates*).  Faces no camera ever sees
   are counted as never seen - the meta reports why each texel exists
   instead of hiding the fallback.

2. **Charts.**  Breadth-first grow connected faces that accept the
   seed's view as a candidate and whose normals stay within 40 degrees
   of the seed: near-planar, single-view patches that project without
   tearing.  Joining on the candidate set (instead of each triangle's
   exact per-face winner) is what keeps a flat panel in one chart -
   near-tie views otherwise shatter it into thousands of single-face
   charts and thousands of seams.

3. **Shelf packing.**  Each chart's rectangle is its bounding box in the
   source view; a uniform scale (capped at 1, area-derived, shrunk until
   everything fits) maps the charts into ``texture_size`` with 2 texels
   of padding.  Ordering is fully deterministic (height, width, index).

4. **Barycentric bake.**  Every chart triangle is rasterized into its
   packed rectangle; each texel's 3D position comes from barycentric
   interpolation, is re-projected into the dominant view, and samples
   that frame bilinearly, again gated by the depth map.  Unsampied texels
   (seams, holes) are dilated inwards for a few passes so linear
   filtering never reads empty space, and faces no camera could ever see
   get a small fallback cell coloured from their own vertices.

Self-checks (both in the meta, like the dense stage's ``sparse_check``):

- ``mapping_check`` - each sampled face's centroid texel is compared
  against a fresh sample of the very frame that painted it.  The bake
  samples at the 3D point's projection while the UV maps through the
  chart bounding box, so this validates the bbox -> pack -> UV chain.
- ``cross_view_check`` - the same texels re-projected into *other*
  views, gated exactly like the bake (front-facing, in frame,
  depth-consistent) so an occluded corner is skipped instead of
  blaming the bake for a correct pixel.  On the synthetic set lighting
  is world-fixed, so the delta measures bake correctness rather than
  exposure changes.

Outputs: ``texture/texture.png``, ``texture/textured.obj`` (per-corner
UVs, so chart seams need no vertex splitting), ``texture/textured.mtl``,
``texture/texture_meta.json``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ReconConfig, ReconstructionError
from .dense import (
    _cv2,
    _depth_path,
    _frames_by_name,
    _load_color,
    _load_depth,
    load_views,
)
from .ply import read_ply_mesh
from .state import StageContext, StateStore, memory_pressure

# Texels left empty between charts; dilation fills them so bilinear
# filtering at a chart border never reads black.
_PADDING = 2
# A chart joins faces whose normals stay this close to its seed (degrees).
_CHART_MAX_DEG = 40.0
# Inward fill passes for unsampled texels (covers the padding).
_DILATE_PASSES = 4
# Corner cell (texels) for faces no camera could ever see.
_NO_DATA = 4
# Shelf-pack scale retries before giving up.
_PACK_RETRIES = 12
# A neighbouring face may join a chart whose view is this close to its
# own best projection (flat panels pick near-tie views per triangle,
# which otherwise shatters them into thousands of single-face charts).
# 0.6 still shattered close-up walkarounds: a receding panel drops
# below 60 % of its best area within a few faces, splitting one flat
# side into hundreds of ~10-face charts painted from alternating frames
# (visible patchwork).  0.25 keeps grazing views out (a face is never
# painted from a view covering less than a quarter of its own best
# projection, ~75 deg) while whole panels unify per view zone.
_VIEW_MARGIN = 0.25
# Cross-view self-check budget (faces, other views per face).
_CHECK_CORNERS = 400
_CHECK_VIEWS = 4


def _project(view: Dict[str, Any], K: np.ndarray, verts: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World -> pixel + depth for every vertex of one view/K."""
    cam = view["R"] @ verts.T.astype(np.float64) + view["t"][:, None]
    z = cam[2]
    front = z > 1e-6
    safe = np.where(front, z, 1.0)
    u = K[0, 0] * cam[0] / safe + K[0, 2]
    v = K[1, 1] * cam[1] / safe + K[1, 2]
    return u, v, z


def _select_views(cfg: ReconConfig, views: List[Dict[str, Any]],
                  depth_paths: Dict[str, Path], verts: np.ndarray,
                  faces: np.ndarray, voxel: float, ctx: StageContext
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dominant view index per face (-1 = seen by no camera).

    Returns ``(face_view, candidates, depth_ok, has_depth, bases, areas,
    area_ref)``: ``candidates[v, f]`` marks the views face ``f`` may
    legitimately be painted from (passes projection and depth gates, and
    is within ``_VIEW_MARGIN`` of the face's own best area) - charts
    join on that set so flat panels do not shatter into per-triangle
    views.  ``bases``/``areas``/``area_ref`` feed chart view election.
    """
    corners = verts[faces]
    n = np.cross(corners[:, 1] - corners[:, 0],
                 corners[:, 2] - corners[:, 0])
    unit = n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    centroids = corners.mean(axis=1)
    n_faces = len(faces)
    n_views = len(views)

    bases = np.zeros((n_views, n_faces), dtype=bool)
    depths_ok = np.zeros((n_views, n_faces), dtype=bool)
    areas = np.zeros((n_views, n_faces), dtype=np.float32)
    has_depth = np.zeros(n_views, dtype=bool)

    # Two quality tiers: depth-checked > projection-only (front-facing,
    # fully in frame, >= 1 px2).  Faces failing both are never seen.
    best_view = {t: np.full(n_faces, -1, dtype=np.int16)
                 for t in ("d", "p")}
    best_area = {t: np.zeros(n_faces, dtype=np.float32)
                 for t in ("d", "p")}

    for view in views:
        vi = view["_index"]
        native = (int(view["width"]), int(view["height"]))
        K_native = view["K"]
        eye = view["center"]
        facing = (unit * (eye[None, :] - centroids)).sum(axis=1) > 0

        u, v, z = _project(view, K_native, verts)
        z3 = z[faces]
        all_front = (z3 > 1e-6).all(axis=1)
        u3, v3 = u[faces], v[faces]
        with np.errstate(invalid="ignore"):
            area = (0.5 * np.abs((u3[:, 1] - u3[:, 0])
                                 * (v3[:, 2] - v3[:, 0])
                                 - (u3[:, 2] - u3[:, 0])
                                 * (v3[:, 1] - v3[:, 0]))).astype(np.float32)
        inside = ((u3 >= 0) & (u3 < native[0])
                  & (v3 >= 0) & (v3 < native[1])).all(axis=1)
        # Close-up walkarounds often frame the subject past the image edge,
        # which made the best face-on view fail an all-vertices-in-frame
        # test and drop the face to an oblique fallback (scrambled livery).
        # The bake already clips per texel and dilates, so a face qualifies
        # when its centroid is inside; out-of-frame texels are still skipped.
        cu = u3.mean(axis=1)
        cv_ = v3.mean(axis=1)
        centroid_in = ((cu >= 0) & (cu < native[0])
                       & (cv_ >= 0) & (cv_ < native[1]))
        base = all_front & facing & centroid_in & (area > 1.0)
        bases[vi] = base
        areas[vi] = area
        if not base.any():
            ctx.check()
            continue

        depth_ok = np.zeros(n_faces, bool)
        path = depth_paths.get(view["name"])
        if path is not None:
            data = _load_depth(path)
            if data is not None:
                depth = data["depth"].astype(np.float32, copy=False)
                valid = data["valid"].astype(bool)
                wh, ww = depth.shape
                has_depth[vi] = True
                # Working size is a uniform scale of native, so native
                # pixel coords map onto the depth map by simple scaling.
                cu = u3.mean(axis=1) * (ww / native[0])
                cv = v3.mean(axis=1) * (wh / native[1])
                cz = z3.mean(axis=1)
                ui = np.rint(cu).astype(np.int64)
                vi_y = np.rint(cv).astype(np.int64)
                near = ((ui >= 0) & (ui < ww) & (vi_y >= 0) & (vi_y < wh)
                        & (cz > 1e-6))
                band = np.maximum(cfg.mesh_trunc_voxels * voxel,
                                  cfg.mesh_trunc_rel * np.maximum(cz, 1e-6))
                ui_c = np.clip(ui, 0, ww - 1)
                vi_c = np.clip(vi_y, 0, wh - 1)
                depth_ok = (near & valid[vi_c, ui_c]
                            & (np.abs(depth[vi_c, ui_c] - cz) <= band))
                depths_ok[vi] = depth_ok

        for tier, ok in (("d", base & depth_ok), ("p", base)):
            better = ok & (area > best_area[tier])
            if better.any():
                best_view[tier][better] = vi
                best_area[tier][better] = area[better]
        ctx.check()

    chosen = np.full(n_faces, -1, dtype=np.int16)
    for tier in ("d", "p"):
        take = (chosen < 0) & (best_view[tier] >= 0)
        chosen[take] = best_view[tier][take]

    # Candidate views per face: within _VIEW_MARGIN of the face's own
    # best passing area, projection-OK, and depth-OK where a depth map
    # exists (occluded pixels never get painted, even from a near tie).
    # A face that fails the depth gate everywhere is painted
    # projection-only regardless, so it is not held to a gate its own
    # tier does not apply either - otherwise it could never join a chart.
    area_ref = np.zeros(n_faces, dtype=np.float32)
    for tier in ("d", "p"):
        m = chosen == best_view[tier]
        area_ref[m] = best_area[tier][m]
    painted_unchecked = ~((bases & depths_ok).any(axis=0))
    candidates = bases & (areas >= _VIEW_MARGIN * area_ref[None, :])
    candidates &= (depths_ok | ~has_depth[:, None]
                   | painted_unchecked[None, :])
    return (chosen, candidates, depths_ok, has_depth,
            bases, areas, area_ref)


def _build_charts(faces: np.ndarray, view: np.ndarray, normals: np.ndarray,
                  candidates: np.ndarray
                  ) -> Tuple[List[np.ndarray], np.ndarray]:
    """BFS charts: candidate view of the seed + normal coherence.

    Joining on ``candidates[seed_view, nb]`` (rather than requiring the
    neighbour's own per-face best to match exactly) is what keeps a flat
    panel in one chart: its triangles rank the near-tie views in
    slightly different orders and would otherwise each pick their own.
    """
    edge_map: Dict[Tuple[int, int], List[int]] = {}
    for fi, f in enumerate(faces):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (int(a), int(b)) if a < b else (int(b), int(a))
            edge_map.setdefault(key, []).append(fi)
    cos_thr = math.cos(math.radians(_CHART_MAX_DEG))
    assigned = np.full(len(faces), -1, dtype=np.int32)
    charts: List[np.ndarray] = []
    for seed in range(len(faces)):
        if assigned[seed] >= 0 or view[seed] < 0:
            continue
        chart_id = len(charts)
        assigned[seed] = chart_id
        members = [seed]
        queue = [seed]
        seed_view = int(view[seed])
        seed_n = normals[seed]
        while queue:
            fi = queue.pop()
            f = faces[fi]
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                key = (int(a), int(b)) if a < b else (int(b), int(a))
                for nb in edge_map[key]:
                    if assigned[nb] >= 0 or not candidates[seed_view, nb]:
                        continue
                    if float(seed_n @ normals[nb]) < cos_thr:
                        continue
                    assigned[nb] = chart_id
                    members.append(nb)
                    queue.append(nb)
        charts.append(np.array(members, dtype=np.int64))
    return charts, assigned


def _flood_face_views(faces: np.ndarray, face_view: np.ndarray,
                      candidates: np.ndarray,
                      area_ref: np.ndarray) -> np.ndarray:
    """Per-face paint view by quality-ordered anchor flooding.

    Per-face max-area ranking flips between near-tie frames along a
    receding panel, so painting each chart from its seed's view left
    hundreds of small patches sourced from alternating frames - a
    patchwork where the livery should read as one surface.  Faces are
    anchored in descending own-best-area order (face-on, most reliable
    spots first); an anchor's frame then floods across shared edges,
    claiming any still-unpainted face for which that frame stays a
    legitimate view (the same ``candidates`` set charts join on).
    View zones collapse to a single frame each, with boundaries only
    where a frame stops being legitimate - not at per-face ranking
    noise.  Never-seen faces stay ``-1``.
    """
    n_faces = len(faces)
    paint = np.full(n_faces, -1, dtype=np.int16)
    owner: Dict[int, List[int]] = {}
    for fi, row in enumerate(faces):
        for raw in row:
            vi = int(raw)
            chain = owner.get(vi)
            if chain is None:
                owner[vi] = [fi]
            else:
                chain.append(fi)
    nbrs: List[List[int]] = [[] for _ in range(n_faces)]
    for chain in owner.values():
        for i in range(len(chain)):
            for j in range(i + 1, len(chain)):
                a, b = chain[i], chain[j]
                if b not in nbrs[a]:
                    nbrs[a].append(b)
                    nbrs[b].append(a)
    del owner

    for raw_f0 in np.argsort(-area_ref, kind="stable"):
        f0 = int(raw_f0)
        if face_view[f0] < 0 or paint[f0] >= 0:
            continue
        view = int(face_view[f0])
        paint[f0] = view
        stack = [f0]
        while stack:
            fi = stack.pop()
            for nb in nbrs[fi]:
                if paint[nb] >= 0 or face_view[nb] < 0:
                    continue
                if candidates[view, nb]:
                    paint[nb] = view
                    stack.append(nb)
    return paint


def _chart_paint_views(charts: List[np.ndarray], paint: np.ndarray,
                       face_view: np.ndarray, area_ref: np.ndarray
                       ) -> np.ndarray:
    """Chart paint view = its members' majority flood view.

    Ties break toward the view covering the most member area, then the
    lower view index (deterministic).  Charts whose members somehow
    stayed unpainted fall back to the seed's own best view.
    """
    out = np.zeros(len(charts), dtype=np.int16)
    for ci, chart in enumerate(charts):
        p = paint[chart]
        p = p[p >= 0]
        if not len(p):
            out[ci] = max(int(face_view[chart[0]]), 0)
            continue
        vals, counts = np.unique(p, return_counts=True)
        best_weight = None
        best_view = int(vals[0])
        for v, c in zip(vals, counts):
            weight = (int(c), float(area_ref[chart][p == v].sum()), -int(v))
            if best_weight is None or weight > best_weight:
                best_weight = weight
                best_view = int(v)
        out[ci] = best_view
    return out


def _chart_boxes(charts: List[np.ndarray], chart_view: np.ndarray,
                 views: List[Dict[str, Any]], verts: np.ndarray,
                 faces: np.ndarray) -> List[Dict[str, Any]]:
    """Chart bounding box in its elected view's native pixels."""
    boxes: List[Dict[str, Any]] = []
    for ci, chart in enumerate(charts):
        vi = int(chart_view[ci])
        v = views[vi]
        uniq = np.unique(faces[chart].reshape(-1))
        u, vv, _z = _project(v, v["K"], verts[uniq])
        inside = ((u >= 0) & (u < v["width"]) & (vv >= 0)
                  & (vv < v["height"]))
        if inside.any():
            u_i, v_i = u[inside], vv[inside]
        else:                                   # fully outside: clamp span
            u_i, v_i = u, vv
        u0 = max(0.0, float(u_i.min()))
        v0 = max(0.0, float(v_i.min()))
        u1 = min(float(v["width"]), max(float(u_i.max()), u0 + 1.0))
        v1 = min(float(v["height"]), max(float(v_i.max()), v0 + 1.0))
        boxes.append({"view": vi, "u0": u0, "v0": v0,
                      "bw": max(1.0, u1 - u0), "bh": max(1.0, v1 - v0)})
    return boxes


def _pack(boxes: List[Dict[str, Any]], atlas: int
          ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
    """Deterministic shelf packing; returns (scale, [(x, y, w, h)...])."""
    total = sum(b["bw"] * b["bh"] for b in boxes)
    usable = float(atlas * atlas)
    # Area-derived scale, but never wider/taller than the atlas itself
    # (a thin or small total can otherwise leave an oversized chart
    # hanging over the edge, which would push UVs past 1.0).
    max_dim = max(max(b["bw"], b["bh"]) for b in boxes)
    scale = min(1.0, math.sqrt(0.85 * usable / max(total, 1e-6)),
                atlas / max(max_dim, 1e-6))
    for _attempt in range(_PACK_RETRIES):
        cells = []
        for i, b in enumerate(boxes):
            w = int(math.ceil(b["bw"] * scale))
            h = int(math.ceil(b["bh"] * scale))
            cells.append((max(1, w), max(1, h), i))
        order = sorted(cells, key=lambda c: (-c[1], -c[0], c[2]))
        placements: List[Optional[Tuple[int, int, int, int]]] = [None] * len(boxes)
        x = y = shelf_h = 0
        fits = True
        for w, h, i in order:
            if x + w > atlas:
                x = 0
                y += shelf_h
                shelf_h = 0
            if y + h > atlas:
                fits = False
                break
            placements[i] = (x, y, w, h)
            x += w + _PADDING
            shelf_h = max(shelf_h, h)
        if fits and all(p is not None for p in placements):
            return scale, [p for p in placements if p is not None]
        scale *= 0.8
    raise ReconstructionError(
        f"Could not pack {len(boxes)} charts into a {atlas}^2 atlas.",
        suggestion="Raise RECON_TEXTURE_SIZE, or lower RECON_TARGET_TRIANGLES.",
        details={"charts": len(boxes), "atlas": atlas},
    )


def _write_obj(path: Path, mtl_name: str, verts: np.ndarray,
               faces: np.ndarray, uv: np.ndarray) -> None:
    """OBJ with per-corner UVs (uv is (F, 3, 2), already v-flipped)."""
    lines = ["mtllib " + mtl_name, "o vehicle"]
    lines.extend(f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts)
    flat = uv.reshape(-1, 2)
    lines.extend(f"vt {a:.5f} {b:.5f}" for a, b in flat)
    lines.append("usemtl vehicle")
    for fi, f in enumerate(faces):
        a, b, c = 3 * fi + 1, 3 * fi + 2, 3 * fi + 3
        lines.append(f"f {f[0]+1}/{a} {f[1]+1}/{b} {f[2]+1}/{c}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dilate(img: np.ndarray, filled: np.ndarray, passes: int) -> int:
    """Fill empty texels from filled 4-neighbours (seams); returns count.

    Any filled neighbour contributes (mean of the filled ones): a 2-texel
    padding column between charts has unfilled texels above/below it, so
    requiring all four neighbours would never fill the seam at all.
    """
    added = 0
    for _ in range(passes):
        nb = (filled[:-2, 1:-1].astype(np.uint8) + filled[2:, 1:-1]
              + filled[1:-1, :-2] + filled[1:-1, 2:])
        new = (~filled[1:-1, 1:-1]) & (nb > 0)
        if not new.any():
            break
        acc = (img[:-2, 1:-1].astype(np.float32) * filled[:-2, 1:-1, None]
               + img[2:, 1:-1] * filled[2:, 1:-1, None]
               + img[1:-1, :-2] * filled[1:-1, :-2, None]
               + img[1:-1, 2:] * filled[1:-1, 2:, None])
        acc /= np.maximum(nb, 1)[..., None]
        region = img[1:-1, 1:-1]
        region[new] = acc[new].astype(np.uint8)
        filled[1:-1, 1:-1] |= new
        added += int(new.sum())
    return added


def _sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray
                     ) -> np.ndarray:
    """Bilinear RGB sample at float pixel coords (clamped)."""
    h, w = img.shape[:2]
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _rects_for(assigned: np.ndarray, placements: List[Tuple[int, int, int, int]]
               ) -> np.ndarray:
    """Per-face packed rect (F, 4) for chart-owned faces (-1 rows elsewhere)."""
    rects = np.full((len(assigned), 4), -1, dtype=np.int64)
    placed = np.asarray(placements, dtype=np.int64)
    ok = assigned >= 0
    rects[ok] = placed[assigned[ok]]
    return rects


def _centroid_texel(uv_px: np.ndarray, fi: int, rect: np.ndarray
                    ) -> Optional[Tuple[int, int]]:
    """Rounded centroid texel of a face, but only if it stays inside the
    face's chart rect (outside lies the dilated padding, which blends
    neighbouring charts and would poison the comparison)."""
    tx, ty = uv_px[fi].mean(axis=0)
    px, py, cw, ch = rect
    if px < 0 or cw < 1:
        return None
    ix, iy = int(round(tx)), int(round(ty))
    if not (px <= ix < px + cw and py <= iy < py + ch):
        return None
    return ix, iy


def _mapping_check(verts: np.ndarray, faces: np.ndarray, paint_view: np.ndarray,
                   uv_px: np.ndarray, rects: np.ndarray, atlas: np.ndarray,
                   atlas_filled: np.ndarray, views: List[Dict[str, Any]],
                   frames: Dict[str, Path], budget: int, ctx: StageContext
                   ) -> Dict[str, Any]:
    """Frame -> atlas round trip: each face's centroid texel is compared
    against a fresh bilinear sample of the very frame that painted it.

    The bake samples the frame at the 3D point's projection while the
    OBJ/UV maps through the chart bounding box, so this validates the
    whole bbox -> pack -> UV chain independently of the other views.
    """
    seen = np.flatnonzero(paint_view >= 0)
    if not len(seen):
        return {"samples": 0}
    stride = max(1, len(seen) // max(1, budget))
    chosen = seen[::stride][:budget]
    centroids = verts[faces].mean(axis=1)
    groups: Dict[int, List[int]] = {}
    for fi in chosen:
        groups.setdefault(int(paint_view[fi]), []).append(fi)
    deltas: List[float] = []
    for vi in sorted(groups):
        view = views[vi]
        path = frames.get(view["name"])
        if path is None:
            continue
        img = _load_color(path, (int(view["width"]), int(view["height"])))
        for fi in groups[vi]:
            texel = _centroid_texel(uv_px, fi, rects[fi])
            if texel is None:
                continue
            ix, iy = texel
            if not atlas_filled[iy, ix]:
                continue
            cam = view["R"] @ centroids[fi] + view["t"]
            if cam[2] <= 1e-6:
                continue
            uu = view["K"][0, 0] * cam[0] / cam[2] + view["K"][0, 2]
            vv = view["K"][1, 1] * cam[1] / cam[2] + view["K"][1, 2]
            if not (0 <= uu < view["width"] - 1 and 0 <= vv < view["height"] - 1):
                continue
            baked = atlas[iy, ix].astype(np.float32)
            real = _sample_bilinear(img, np.array([uu]),
                                    np.array([vv]))[0, ::-1]  # BGR -> RGB
            deltas.append(float(np.abs(baked - real).mean()))
        ctx.check()
    if not deltas:
        return {"samples": 0}
    arr = np.array(deltas)
    return {
        "samples": int(len(arr)),
        "mean_abs_delta_rgb": round(float(arr.mean()), 2),
        "p50_abs_delta_rgb": round(float(np.percentile(arr, 50)), 2),
        "p90_abs_delta_rgb": round(float(np.percentile(arr, 90)), 2),
    }


def _cross_view_check(cfg: ReconConfig, voxel: float,
                      depth_paths: Dict[str, Path],
                      verts: np.ndarray, faces: np.ndarray,
                      normals: np.ndarray, paint_view: np.ndarray,
                      uv_px: np.ndarray, rects: np.ndarray,
                      atlas: np.ndarray, atlas_filled: np.ndarray,
                      views: List[Dict[str, Any]], frames: Dict[str, Path],
                      budget: int, n_views: int, ctx: StageContext
                      ) -> Dict[str, Any]:
    """Centroid texels re-projected into OTHER views and compared there.

    Gated the way the bake was gated: front-facing in that view, inside
    the frame, and depth-consistent with that view's depth map - an
    occluded corner would compare the vehicle against the background
    and blame the bake for a correct pixel.  The own-paint view is
    skipped (its delta is the mapping check).
    """
    seen = np.flatnonzero(paint_view >= 0)
    if not len(seen):
        return {"samples": 0}
    stride = max(1, len(seen) // max(1, budget))
    chosen = seen[::stride][:budget]
    centroids = verts[faces].mean(axis=1)
    other = [i for i in range(0, len(views),
                              max(1, len(views) // n_views))][:n_views]
    deltas: List[float] = []
    occluded = 0
    for vi in other:
        view = views[vi]
        path = frames.get(view["name"])
        if path is None:
            continue
        img = _load_color(path, (int(view["width"]), int(view["height"])))
        depth = valid = None
        wh = ww = 0
        dpath = depth_paths.get(view["name"])
        if dpath is not None:
            data = _load_depth(dpath)
            if data is not None:
                depth = data["depth"].astype(np.float32, copy=False)
                valid = data["valid"].astype(bool)
                wh, ww = depth.shape
        u, v, z = _project(view, view["K"], verts)
        eye = view["center"]
        for fi in chosen:
            if paint_view[fi] == vi:
                continue                    # own paint view: mapping check
            if float(normals[fi] @ (eye - centroids[fi])) <= 0:
                continue
            texel = _centroid_texel(uv_px, fi, rects[fi])
            if texel is None:
                continue
            ix, iy = texel
            if not atlas_filled[iy, ix]:
                continue
            cam = view["R"] @ centroids[fi] + view["t"]
            if cam[2] <= 1e-6:
                continue
            uu = view["K"][0, 0] * cam[0] / cam[2] + view["K"][0, 2]
            vv = view["K"][1, 1] * cam[1] / cam[2] + view["K"][1, 2]
            if not (0 <= uu < view["width"] - 1 and 0 <= vv < view["height"] - 1):
                continue
            if depth is not None:
                ui = int(round(uu * ww / view["width"]))
                vj = int(round(vv * wh / view["height"]))
                band = np.maximum(cfg.mesh_trunc_voxels * voxel,
                                  cfg.mesh_trunc_rel * cam[2])
                if not (0 <= ui < ww and 0 <= vj < wh) or not valid[vj, ui] \
                        or abs(float(depth[vj, ui]) - cam[2]) > band:
                    occluded += 1
                    continue
            baked = atlas[iy, ix].astype(np.float32)
            real = _sample_bilinear(img, np.array([uu]),
                                    np.array([vv]))[0, ::-1]  # BGR -> RGB
            deltas.append(float(np.abs(baked - real).mean()))
        ctx.check()
    if not deltas:
        return {"samples": 0, "occluded_skipped": occluded}
    arr = np.array(deltas)
    return {
        "samples": int(len(arr)),
        "occluded_skipped": int(occluded),
        "mean_abs_delta_rgb": round(float(arr.mean()), 2),
        "p50_abs_delta_rgb": round(float(np.percentile(arr, 50)), 2),
        "p90_abs_delta_rgb": round(float(np.percentile(arr, 90)), 2),
    }


def run_texture(cfg: ReconConfig, store: StateStore,
                ctx: StageContext) -> Dict[str, Any]:
    src = cfg.d("lowpoly", "lowpoly.ply")
    if not src.is_file():
        raise ReconstructionError(
            "Texture stage needs the low-poly mesh (lowpoly/lowpoly.ply).",
            suggestion="Run the pipeline through the 'lowpoly' stage first.",
            details={"project": str(cfg.project_dir)},
        )
    out_dir = cfg.d("texture")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "texture.png"
    obj_path = out_dir / "textured.obj"
    mtl_path = out_dir / "textured.mtl"
    meta_path = out_dir / "texture_meta.json"

    atlas_size = max(256, min(4096, int(cfg.texture_size)))
    ctx.set_command(
        f"hand-rolled UV atlas + bake to {atlas_size}^2 (per-view "
        f"dominant projection, shelf packing)")
    ctx.log(f"baking texture atlas {atlas_size}^2 for "
            f"{src.relative_to(cfg.project_dir)}")
    started = time.time()
    ctx.check()

    verts, rgb, faces = read_ply_mesh(src)
    if len(faces) < 4 or len(verts) < 4:
        raise ReconstructionError(
            f"Low-poly mesh is too small to texture ({len(verts)} vertices, "
            f"{len(faces)} faces).",
            suggestion="Inspect lowpoly/lowpoly_meta.json; the upstream "
                       "mesh or decimation stage produced a fragment.",
            details={"vertices": int(len(verts)), "faces": int(len(faces))},
        )
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)

    # Voxel size: the depth gate uses the same band the mesh fused with.
    voxel = 0.0
    mesh_meta = cfg.d("mesh", "mesh_meta.json")
    if mesh_meta.is_file():
        try:
            voxel = float(json.loads(
                mesh_meta.read_text(encoding="utf-8"))["volume"]["voxel"])
        except (OSError, ValueError, KeyError, TypeError):
            voxel = 0.0
    if voxel <= 0:
        voxel = 0.01

    views = load_views(cfg)
    for index, view in enumerate(views):
        view["_index"] = index
    frames = _frames_by_name(cfg)
    depth_paths = {v["name"]: _depth_path(cfg, v["name"]) for v in views}
    missing = [v["name"] for v in views if v["name"] not in frames]
    if missing:
        raise ReconstructionError(
            f"Texture stage found no frame for {len(missing)} views.",
            suggestion="Re-run frame selection/extraction; selected frames "
                       "must cover the registered views.",
            details={"missing": missing[:5]},
        )

    face_view, candidates, depths_ok, has_depth, bases, areas, area_ref = (
        _select_views(cfg, views, depth_paths, verts, faces, voxel, ctx))
    if (face_view >= 0).sum() == 0:
        raise ReconstructionError(
            "No face is visible from any registered camera.",
            suggestion="The solve does not observe this geometry; check "
                       "reconstruction_report.json and the sparse previews.",
            details={"project": str(cfg.project_dir)},
        )

    corners = verts[faces]
    n = np.cross(corners[:, 1] - corners[:, 0],
                 corners[:, 2] - corners[:, 0])
    normals = n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    charts, assigned = _build_charts(faces, face_view, normals, candidates)
    if not charts:
        raise ReconstructionError(
            "No texture chart could be grown from the visible faces.",
            suggestion="Check reconstruction_report.json and re-run the "
                       "dense stage.",
            details={"project": str(cfg.project_dir)},
        )

    # The view that actually paints a face is its chart's majority view
    # from the quality-ordered face flood: adjacent charts on the same
    # panel converge on one frame per view zone instead of alternating
    # frames per small chart (which read as a patchwork livery).
    paint_face = _flood_face_views(faces, face_view, candidates, area_ref)
    chart_view = _chart_paint_views(charts, paint_face, face_view,
                                    area_ref)
    paint_view = np.full(len(faces), -1, dtype=np.int16)
    has_chart = assigned >= 0
    paint_view[has_chart] = chart_view[assigned[has_chart]]
    seen = paint_view >= 0
    idx = np.flatnonzero(seen)
    painted_ok = depths_ok[paint_view[idx], idx] & has_depth[paint_view[idx]]
    tiers = {
        "depth_checked": int(painted_ok.sum()),
        "projection_only": int((~painted_ok).sum()),
        "never_seen": int((~seen).sum()),
        "margin_joined": int((paint_view[seen] != face_view[seen]).sum()),
    }
    ctx.note(f"view selection (as painted): {tiers}")
    boxes = _chart_boxes(charts, chart_view, views, verts, faces)
    scale, placements = _pack(boxes, atlas_size)
    ctx.note(f"{len(charts)} charts, pack scale {scale:.3f}")
    ctx.check()

    # Per-face-corner UV in packed texel pixels (for bake + OBJ).
    atlas_px = np.zeros((len(faces), 3, 2), dtype=np.float64)
    uv = np.zeros((len(faces), 3, 2), dtype=np.float32)
    for chart, box, (px, py, cw, ch) in zip(charts, boxes, placements):
        vi = box["view"]
        v = views[vi]
        # _project takes an (N, 3) point array: flatten the (M, 3, 3)
        # corners, project, then regroup per face.
        u_c, v_c, _z = _project(v, v["K"], verts[faces[chart]].reshape(-1, 3))
        u_c = u_c.reshape(-1, 3)
        v_c = v_c.reshape(-1, 3)
        dx = (u_c - box["u0"]) / box["bw"]
        dy = (v_c - box["v0"]) / box["bh"]
        tx = px + np.clip(dx, 0.0, 1.0) * cw
        ty = py + np.clip(dy, 0.0, 1.0) * ch
        atlas_px[chart] = np.stack([tx, ty], axis=-1)
        uv[chart, :, 0] = tx / atlas_size
        uv[chart, :, 1] = 1.0 - ty / atlas_size

    # Fallback cell for faces no camera ever saw (bottom-right corner).
    nd = _NO_DATA
    unseen = paint_view < 0
    if unseen.any():
        uv[unseen] = ((atlas_size - nd / 2.0) / atlas_size,
                      nd / 2.0 / atlas_size)
    cv2 = _cv2()
    atlas_img = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    filled = np.zeros((atlas_size, atlas_size), dtype=bool)

    # Bake chart by chart, view by view (one frame in memory at a time).
    by_view: Dict[int, List[int]] = {}
    for ci, box in enumerate(boxes):
        by_view.setdefault(box["view"], []).append(ci)
    baked_faces = 0
    depth_rejected = 0
    for vi in sorted(by_view):
        view = views[vi]
        img = _load_color(frames[view["name"]],
                          (int(view["width"]), int(view["height"])))
        depth = valid = None
        path = depth_paths.get(view["name"])
        wh = ww = 0
        if path is not None:
            data = _load_depth(path)
            if data is not None:
                depth = data["depth"].astype(np.float32, copy=False)
                valid = data["valid"].astype(bool)
                wh, ww = depth.shape
        native_w = int(view["width"])
        native_h = int(view["height"])
        for ci in by_view[vi]:
            # Packed texel coords were computed into atlas_px already.
            for fi in charts[ci]:
                a, b, c = atlas_px[fi]
                x_lo = max(0, int(np.floor(min(a[0], b[0], c[0]))))
                x_hi = min(atlas_size - 1, int(np.ceil(max(a[0], b[0], c[0]))))
                y_lo = max(0, int(np.floor(min(a[1], b[1], c[1]))))
                y_hi = min(atlas_size - 1, int(np.ceil(max(a[1], b[1], c[1]))))
                if x_hi < x_lo or y_hi < y_lo:
                    continue
                gx, gy = np.meshgrid(np.arange(x_lo, x_hi + 1),
                                     np.arange(y_lo, y_hi + 1))
                denom = ((b[0] - a[0]) * (c[1] - a[1])
                         - (c[0] - a[0]) * (b[1] - a[1]))
                if abs(denom) < 1e-9:
                    continue
                w0 = ((b[0] - gx) * (c[1] - gy)
                      - (c[0] - gx) * (b[1] - gy)) / denom
                w1 = ((c[0] - gx) * (a[1] - gy)
                      - (a[0] - gx) * (c[1] - gy)) / denom
                w2 = 1.0 - w0 - w1
                inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
                if not inside.any():
                    continue
                pts = (w0[..., None] * verts[faces[fi, 0]]
                       + w1[..., None] * verts[faces[fi, 1]]
                       + w2[..., None] * verts[faces[fi, 2]])
                cam = (view["R"] @ pts.reshape(-1, 3).T
                       + view["t"][:, None]).reshape(3, *gx.shape)
                zc = cam[2]
                front = zc > 1e-6
                safe = np.where(front, zc, 1.0)
                uu = view["K"][0, 0] * cam[0] / safe + view["K"][0, 2]
                vv = view["K"][1, 1] * cam[1] / safe + view["K"][1, 2]
                ok = inside & front & (uu >= 0) & (uu < native_w - 1) \
                    & (vv >= 0) & (vv < native_h - 1)
                if depth is not None:
                    pre_depth = ok.copy()
                    du = uu * (ww / native_w)
                    dv = vv * (wh / native_h)
                    ui = np.clip(np.rint(du).astype(np.int64), 0, ww - 1)
                    vi_y = np.clip(np.rint(dv).astype(np.int64), 0, wh - 1)
                    band = np.maximum(cfg.mesh_trunc_voxels * voxel,
                                      cfg.mesh_trunc_rel * np.maximum(zc, 1e-6))
                    ok = (ok & valid[vi_y, ui]
                          & (np.abs(depth[vi_y, ui] - zc) <= band))
                    depth_rejected += int((pre_depth & ~ok).sum())
                if not ok.any():
                    continue
                # Sample every bbox texel (clamped where invalid - those
                # are masked out below), then write only the accepted,
                # not-yet-owned ones; nearest chart wins any overlap.
                color = _sample_bilinear(img, uu.ravel(),
                                         vv.ravel()).reshape(gx.shape + (3,))
                region = atlas_img[y_lo:y_hi + 1, x_lo:x_hi + 1]
                region_f = filled[y_lo:y_hi + 1, x_lo:x_hi + 1]
                take = ok & ~region_f
                region[take] = color[take][..., ::-1]  # BGR -> RGB
                region_f |= ok
                baked_faces += 1
        del img
        ctx.check()

    # Fallback cell colour: mean vertex colour of never-seen faces.
    if unseen.any():
        mean_rgb = rgb[faces[unseen]].reshape(-1, 3).mean(axis=0)
        atlas_img[atlas_size - nd:, atlas_size - nd:] = mean_rgb.astype(np.uint8)
        filled[atlas_size - nd:, atlas_size - nd:] = True

    dilated = _dilate(atlas_img, filled, _DILATE_PASSES)
    if (~filled).any():
        # Anything still empty (interior chart holes) takes the mesh's
        # average colour rather than black.
        atlas_img[~filled] = rgb.mean(axis=0).astype(np.uint8)
    filled_share = round(float(filled.mean()), 4)

    rects = _rects_for(assigned, placements)
    mapping = _mapping_check(verts, faces, paint_view, atlas_px, rects,
                             atlas_img, filled, views, frames,
                             _CHECK_CORNERS, ctx)
    check = _cross_view_check(cfg, voxel, depth_paths, verts, faces, normals,
                              paint_view, atlas_px, rects, atlas_img, filled,
                              views, frames, _CHECK_CORNERS, _CHECK_VIEWS,
                              ctx)

    ctx.check()
    cv2.imwrite(str(png_path), atlas_img[..., ::-1])
    _write_obj(obj_path, "textured.mtl", verts, faces, uv)
    mtl_path.write_text(
        "newmtl vehicle\n"
        "Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\nNs 0\n"
        "illum 1\nmap_Kd texture.png\n",
        encoding="utf-8",
    )
    ctx.add_output_files([png_path, obj_path, mtl_path])

    used_area = sum(w * h for _x, _y, w, h in placements)
    meta: Dict[str, Any] = {
        "fingerprint": cfg.texture_fingerprint(),
        "algorithm": "per-view dominant projection, BFS charts, shelf "
                     "packing, barycentric bake",
        "atlas_size": atlas_size,
        "views_total": len(views),
        "faces": int(len(faces)),
        "view_selection": tiers,
        "charts": len(charts),
        "pack_scale": round(scale, 4),
        "atlas_utilisation": round(used_area / float(atlas_size ** 2), 4),
        "faces_baked": int(baked_faces),
        "depth_rejected_texels": int(depth_rejected),
        "filled_share": filled_share,
        "dilated_texels": int(dilated),
        "unseen_face_share": round(float(unseen.mean()), 4),
        "cross_view_check": check,
        "mapping_check": mapping,
        "output": {
            "png": str(png_path.relative_to(cfg.project_dir)),
            "obj": str(obj_path.relative_to(cfg.project_dir)),
            "mtl": str(mtl_path.relative_to(cfg.project_dir)),
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "memory_peak_gb": memory_pressure(cfg.ram_limit_gb)["process_peak_gb"],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ctx.add_output_files([meta_path])

    ctx.log(
        f"texture: {len(charts)} charts into {atlas_size}^2, "
        f"{filled_share:.0%} filled, mapping delta "
        f"{mapping.get('mean_abs_delta_rgb', float('nan'))}, cross-view "
        f"{check.get('mean_abs_delta_rgb', float('nan'))} "
        f"({meta['elapsed_seconds']}s)",
        operation="complete",
    )
    result = {
        "texture": meta["output"]["png"],
        "obj": meta["output"]["obj"],
        "atlas_size": atlas_size,
        "charts": len(charts),
        "filled_share": filled_share,
        "mapping_check": mapping,
        "cross_view_check": check,
        "elapsed_seconds": meta["elapsed_seconds"],
    }
    ctx.result(result)
    return result
