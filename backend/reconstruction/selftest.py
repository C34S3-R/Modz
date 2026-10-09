"""Small synthetic test dataset with ground truth (section 29).

"Vehicle-like" box rendered from a known orbit so every stage can be
*verified* instead of merely run:

    selftest.py render  <dir> [--views 16]     # images + gt.json
    selftest.py run                              # full pipeline + metrics
    selftest.py evaluate <project>               # re-score an existing run

Verification uses quantities with known ground truth:

  * camera poses after similarity alignment (Umeyama),
  * relative rotation error per image pair (coordinate-free),
  * mean reprojection error,
  * registered-camera ratio.

The rasteriser is intentionally simple (z-buffer, barycentric, textured
triangles) and runs on numpy alone - no OpenGL, no GPU, nothing to install.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Mesh: a box with vehicle-like proportions (matatu-sized, metres)
# ---------------------------------------------------------------------------
def build_box(length: float = 4.2, width: float = 1.8, height: float = 1.6):
    lx, wy, hz = length / 2, width / 2, height
    verts = np.array([
        [-lx, -wy, 0.0], [lx, -wy, 0.0], [lx, wy, 0.0], [-lx, wy, 0.0],
        [-lx, -wy, hz], [lx, -wy, hz], [lx, wy, hz], [-lx, wy, hz],
    ], dtype=np.float64)
    quads = [
        (0, 1, 5, 4),   # -Y side
        (2, 3, 7, 6),   # +Y side
        (3, 0, 4, 7),   # -X (front)
        (1, 2, 6, 5),   # +X (rear)
        (4, 5, 6, 7),   # roof
        (3, 2, 1, 0),   # floor
    ]
    uv_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    tris, tri_uv = [], []
    for quad in quads:
        a, b, c, d = quad
        tris += [(a, b, c), (a, c, d)]
        tri_uv += [uv_corners[[0, 1, 2]], uv_corners[[0, 2, 3]]]
    return verts, np.array(tris, dtype=np.int64), np.array(tri_uv), quads


def make_textures(rng: np.random.Generator, n_faces: int = 6,
                  size: int = 384) -> List[np.ndarray]:
    """Distinct, high-contrast, SIFT-friendly texture per face.

    Real vehicles carry lettering, grilles, dirt, panel lines and stickers,
    so the synthetic faces deliberately mix scales: large panels, mid-size
    marks and a dense layer of small marks.  Without the small scale, SIFT
    finds only a few hundred points per frame, which is exactly the kind of
    input that stalls real reconstructions.
    """
    textures = []
    palette = np.array([
        [210, 60, 50], [40, 90, 200], [240, 200, 40],
        [30, 160, 90], [230, 230, 230], [120, 60, 170],
    ], dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    for face in range(n_faces):
        tex = np.empty((size, size, 3), dtype=np.uint8)
        tex[:] = palette[face % len(palette)]
        # Blocks (windows / panels look)
        for _ in range(26):
            x0 = int(rng.integers(0, size - 40))
            y0 = int(rng.integers(0, size - 40))
            w = int(rng.integers(14, 70))
            h = int(rng.integers(14, 70))
            colour = rng.integers(0, 255, size=3, dtype=np.uint8)
            tex[y0:y0 + h, x0:x0 + w] = colour
        # Circles (rivets / lights)
        for _ in range(24):
            cx = int(rng.integers(20, size - 20))
            cy = int(rng.integers(20, size - 20))
            r = int(rng.integers(5, 22))
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            tex[mask] = rng.integers(0, 255, size=3, dtype=np.uint8)
        # Multi-scale checker strips (panel gaps / grilles)
        for tile in (8, 16, 32):
            checker = (((xx // tile) + (yy // tile)) % 2).astype(bool)
            band = np.zeros((size, size), dtype=bool)
            band[:, :18] = True
            tex[band & checker] = np.array([20, 20, 20], dtype=np.uint8)
            tex[band & ~checker] = np.array([250, 250, 250], dtype=np.uint8)
        # Dense small marks: the fine scale that gives SIFT something to
        # latch onto (text, dirt, bolt heads, decals).
        for _ in range(1400):
            x0 = int(rng.integers(0, size - 7))
            y0 = int(rng.integers(0, size - 7))
            w = int(rng.integers(3, 7))
            h = int(rng.integers(3, 7))
            tex[y0:y0 + h, x0:x0 + w] = rng.integers(0, 255, size=3,
                                                      dtype=np.uint8)
        # Short strokes (lettering-like runs of 2-3 blobs)
        for _ in range(120):
            x0 = int(rng.integers(0, size - 30))
            y0 = int(rng.integers(0, size - 8))
            w = int(rng.integers(8, 30))
            tex[y0:y0 + 4, x0:x0 + w] = rng.integers(0, 255, size=3,
                                                      dtype=np.uint8)
        textures.append(tex)
    return textures


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
def look_at(eye: np.ndarray, target: np.ndarray,
            world_up: np.ndarray = np.array([0.0, 0.0, 1.0])
            ) -> Tuple[np.ndarray, np.ndarray]:
    """World->camera rotation/translation with camera +Z looking at target."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-12)
    down = np.cross(forward, right)          # image +y direction
    R = np.vstack([right, down, forward])    # rows = camera axes
    t = -R @ eye
    return R, t


def orbit_cameras(n_views: int, rng: np.random.Generator,
                  radius: float = 8.0, target=(0.0, 0.0, 0.9)):
    views = []
    target = np.array(target, dtype=np.float64)
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views + float(rng.uniform(-0.03, 0.03))
        height = 1.6 + 1.1 * (0.5 + 0.5 * math.sin(theta * 2.0)) \
            + float(rng.uniform(-0.1, 0.1))
        eye = np.array([radius * math.cos(theta),
                        radius * math.sin(theta), height])
        R, t = look_at(eye, target)
        views.append({"eye": eye, "R": R, "t": t})
    return views


# ---------------------------------------------------------------------------
# Rasteriser
# ---------------------------------------------------------------------------
def render_view(verts, tris, tri_uv, textures, R, t, K, width, height,
                rng: Optional[np.random.Generator] = None,
                backdrop: int = 26) -> np.ndarray:
    cam = (verts @ R.T) + t
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    px = K[0, 0] * cam[:, 0] / safe_z + K[0, 2]
    py = K[1, 1] * cam[:, 1] / safe_z + K[1, 2]

    image = np.full((height, width, 3), backdrop, dtype=np.uint8)
    zbuf = np.full((height, width), np.inf, dtype=np.float64)

    for tri_index, (i0, i1, i2) in enumerate(tris):
        zz = z[[i0, i1, i2]]
        if (zz <= 0.05).any():
            continue
        # Back-face culling (camera sits at the origin in camera space).
        n = np.cross(cam[i1] - cam[i0], cam[i2] - cam[i0])
        if np.dot(n, -(cam[i0] + cam[i1] + cam[i2]) / 3.0) <= 0:
            continue
        p = np.column_stack([px[[i0, i1, i2]], py[[i0, i1, i2]]])
        min_x = max(int(math.floor(p[:, 0].min())), 0)
        max_x = min(int(math.ceil(p[:, 0].max())), width - 1)
        min_y = max(int(math.floor(p[:, 1].min())), 0)
        max_y = min(int(math.ceil(p[:, 1].max())), height - 1)
        if min_x > max_x or min_y > max_y:
            continue
        xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        v0 = p[1] - p[0]
        v1 = p[2] - p[0]
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-9:
            continue
        dx = gx - p[0, 0]
        dy = gy - p[0, 1]
        b1 = (dx * v1[1] - v1[0] * dy) / den
        b2 = (v0[0] * dy - dx * v0[1]) / den
        b0 = 1.0 - b1 - b2
        mask = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
        if not mask.any():
            continue
        # Perspective-correct depth and texture coordinates.
        inv_z = b0 / zz[0] + b1 / zz[1] + b2 / zz[2]
        depth = 1.0 / np.where(inv_z > 1e-9, inv_z, 1e-9)
        sub_z = zbuf[min_y:max_y + 1, min_x:max_x + 1]
        take = mask & (depth < sub_z)
        if not take.any():
            continue
        uv = tri_uv[tri_index]
        u = (b0 * uv[0, 0] / zz[0] + b1 * uv[1, 0] / zz[1] +
             b2 * uv[2, 0] / zz[2]) / np.where(inv_z > 1e-9, inv_z, 1e-9)
        v = (b0 * uv[0, 1] / zz[0] + b1 * uv[1, 1] / zz[1] +
             b2 * uv[2, 1] / zz[2]) / np.where(inv_z > 1e-9, inv_z, 1e-9)
        tex = textures[tri_index // 2]
        size = tex.shape[0]
        ix = np.clip((u * (size - 1)).astype(np.int32), 0, size - 1)
        iy = np.clip((v * (size - 1)).astype(np.int32), 0, size - 1)
        colour = tex[iy, ix]
        window = image[min_y:max_y + 1, min_x:max_x + 1]
        window[take] = colour[take]
        sub_z[take] = depth[take]

    if rng is not None:
        noise = rng.normal(0.0, 1.6, size=image.shape)
        image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return image


# ---------------------------------------------------------------------------
def render_dataset(out_dir: Path, n_views: int = 48,
                   width: int = 640, height: int = 480,
                   focal: Optional[float] = None,
                   fov_degrees: float = 65.0, seed: int = 5) -> Dict[str, Any]:
    """Write frames + gt.json into ``out_dir``; returns the GT payload.

    Defaults imitate a phone walk-around: ~65 degrees horizontal field of
    view (what ``ReconConfig`` assumes for video without EXIF) and a
    ~7.5 degree step per view, which is what a person circling a vehicle
    produces at the default 0.5 s frame interval.  A coarser orbit would
    make the synthetic set harder than any real recording and would hide
    genuine regressions behind a test artefact.
    """
    import cv2  # noqa: F401  (kept honest: JPEG writing via OpenCV)

    if focal is None or focal <= 0:
        focal = (width / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("view_*.jpg"):
        old.unlink()
    rng = np.random.default_rng(seed)
    verts, tris, tri_uv, _quads = build_box()
    textures = make_textures(rng)
    K = np.array([[focal, 0.0, width / 2.0],
                  [0.0, focal, height / 2.0],
                  [0.0, 0.0, 1.0]])
    views = orbit_cameras(n_views, rng)

    entries = []
    for index, view in enumerate(views, start=1):
        image = render_view(verts, tris, tri_uv, textures,
                            view["R"], view["t"], K, width, height, rng)
        name = f"view_{index:03d}.jpg"
        path = out_dir / name
        ok, buf = cv2.imencode(".jpg", image,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"could not encode {name}")
        buf.tofile(str(path))
        entries.append({
            "file": name,
            "R": [float(x) for x in view["R"].ravel()],
            "t": [float(x) for x in view["t"].ravel()],
            "center": [float(x) for x in view["eye"]],
        })

    gt = {
        "width": width, "height": height, "focal": float(focal),
        "principal_point": [width / 2.0, height / 2.0],
        "camera_count": len(entries),
        "scene_size": 4.2,
        "images": entries,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "gt.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    return gt


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def rotation_angle_deg(R: np.ndarray) -> float:
    trace = float(np.trace(R)) - 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, trace / 2.0))))


def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (scale, rotation, translation) src -> dst."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    Rm = U @ D @ Vt
    var_s = (xs ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(S) @ D) / (var_s + 1e-12))
    t = mu_d - scale * Rm @ mu_s
    return scale, Rm, t


def _gt_name_map(project_dir: Path, gt: Dict[str, Any]) -> Dict[str, Any]:
    """Map reconstructed frame names back to the rendered source images.

    Frames are renamed during extraction (``view_007.jpg`` ->
    ``frame_000007.jpg``), and the renaming must stay reversible or the
    comparison silently matches nothing.  The extraction manifest is the
    authority; a position-based fallback covers older projects.
    """
    gt_by_source = {e["file"].rsplit(".", 1)[0]: e for e in gt["images"]}
    manifest = project_dir / "frames" / "extraction_manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            pairs = []
            for entry in data.get("frames", []):
                source = entry.get("source")
                if source is None:
                    return _fallback_name_map(project_dir, gt)
                pairs.append((Path(entry["file"]).stem,
                              source.rsplit(".", 1)[0]))
            if pairs:
                return {frame: gt_by_source[source]
                        for frame, source in pairs
                        if source in gt_by_source}
        except (OSError, ValueError):
            pass
    return _fallback_name_map(project_dir, gt)


def _fallback_name_map(project_dir: Path, gt: Dict[str, Any]) -> Dict[str, Any]:
    """Position-based mapping: extraction preserves the input order."""
    original = project_dir / "frames" / "original"
    frames = sorted(p.stem for p in original.glob("frame_*")) \
        if original.is_dir() else []
    sources = sorted(gt["images"], key=lambda e: e["file"])
    return {frame: entry for frame, entry in zip(frames, sources)}


def evaluate_mesh(project_dir: Path, gt: Dict[str, Any],
                  alignment: Tuple[float, np.ndarray, np.ndarray]
                  ) -> Dict[str, Any]:
    """Score ``mesh/raw/mesh.ply`` (and ``lowpoly/lowpoly.ply`` once the
    decimation stage has run) against the synthetic GT box.

    The meshes live in the estimated SfM frame; the same similarity that
    aligns estimated camera centres to GT (Umeyama) maps them back, so the
    meshes can be compared to the box in GT coordinates directly.  The box
    signed distance is exact:  max(|p - c|) - half_size  per axis.
    """
    mesh_path = Path(project_dir) / "mesh" / "raw" / "mesh.ply"
    if not mesh_path.is_file():
        return {"present": False}
    try:
        from .ply import read_ply_mesh
    except ImportError:                       # run as a plain script
        sys.path.insert(0, str(BACKEND_DIR))
        from reconstruction.ply import read_ply_mesh
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        return {"present": True, "error": f"scipy required: {exc}"}

    scale, Rm, tvec = alignment

    length = float(gt.get("scene_size", 4.2))   # build_box() defaults
    half = np.array([length / 2.0,
                     length * (1.8 / 4.2) / 2.0,
                     length * (1.6 / 4.2) / 2.0])
    centre = np.array([0.0, 0.0, half[2]])      # box floor sits at z = 0

    # One dense sample of the GT surface, shared by every mesh scored
    # here: a mesh that never formed (or formed elsewhere) fails coverage
    # even if its own bounding box looks plausible.
    n = 21
    a = np.linspace(-half[0], half[0], n)
    b = np.linspace(-half[1], half[1], n)
    c = np.linspace(0.0, 2 * half[2], n)
    ab = np.meshgrid(a, b, indexing="ij")       # z faces
    bc = np.meshgrid(b, c, indexing="ij")       # x faces
    ac = np.meshgrid(a, c, indexing="ij")       # y faces
    flat = lambda g: (g[0].ravel(), g[1].ravel())          # noqa: E731
    f0x, f0y = flat(ab)
    f2y, f2z = flat(bc)
    f4x, f4z = flat(ac)
    face0 = np.stack([f0x, f0y, np.zeros(n * n)], axis=1)    # z = 0
    face1 = np.stack([f0x, f0y, np.full(n * n, 2 * half[2])],
                     axis=1)                                 # z = h
    face2 = np.stack([np.full(n * n, -half[0]), f2y, f2z],
                     axis=1)                                 # x = -l/2
    face3 = np.stack([np.full(n * n, half[0]), f2y, f2z],
                     axis=1)                                 # x = +l/2
    face4 = np.stack([f4x, np.full(n * n, -half[1]), f4z],
                     axis=1)                                 # y = -w/2
    face5 = np.stack([f4x, np.full(n * n, half[1]), f4z],
                     axis=1)                                 # y = +w/2
    # Only sample surface the GT cameras could actually see: the box is
    # convex (no occluders), so visibility per face is exact.  The orbit
    # has a blind rear (the pipeline reports it) and a floor no camera
    # looks at - penalising the reconstruction for *not inventing* those
    # faces would punish honesty.  Unseen faces are reported instead.
    blocks = [face0, face1, face2, face3, face4, face5]
    face_names = ["floor", "top", "x_minus", "x_plus", "y_minus", "y_plus"]
    normals = [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0),
               (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)]
    cams = np.array([np.asarray(e["center"], dtype=np.float64)
                     for e in gt.get("images", [])]) \
        if gt.get("images") else np.zeros((0, 3))
    visible, unseen_faces = [], []
    for block, normal, name in zip(blocks, normals, face_names):
        n = np.asarray(normal)
        if not len(cams) or np.any((cams - block[0]) @ n > 0):
            visible.append(block)
        else:
            unseen_faces.append(name)
    samples = np.concatenate(visible if visible else blocks)

    def score(verts: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        """Metrics + pass/fail for one mesh in the estimated frame."""
        verts_gt = ((verts - tvec) @ Rm) / scale  # estimated -> GT frame
        q = np.abs(verts_gt - centre) - half
        sd = q.max(axis=1)   # exact signed distance to the GT box
        lo, hi = np.percentile(verts_gt, [0.5, 99.5], axis=0)
        dims = hi - lo
        gt_dims = half * 2.0
        dim_err = np.abs(dims - gt_dims) / gt_dims
        nearest, _ = cKDTree(verts_gt).query(samples)
        junk_share = float(np.mean(np.abs(sd) > 0.15))
        m = {
            "vertices": int(len(verts)),
            "dimensions": [round(float(x), 3) for x in dims],
            "gt_dimensions": [round(float(x), 3) for x in gt_dims],
            "dimension_rel_error": [round(float(x), 4) for x in dim_err],
            "surface_dist_p50": round(float(np.percentile(np.abs(sd), 50)), 4),
            "surface_dist_p90": round(float(np.percentile(np.abs(sd), 90)), 4),
            "junk_share_gt_015": round(junk_share, 4),
            "coverage_nearest_p50": round(float(np.median(nearest)), 4),
            "coverage_nearest_p95": round(float(np.percentile(nearest, 95)),
                                          4),
        }
        ok = {
            # Calibration point (see README "Verification"): these four
            # thresholds were chosen against measured outcomes on both
            # verification sets.  Each sits >= 1.9x away from the nearest
            # known-bad variant (ungated synth: dims +158..+695 %,
            # surface p90 3.20, junk 0.97; support-fragment synth:
            # coverage p95 1.27) while leaving >= 1.3x headroom over the
            # three-gate measurements (dims 0.133, surface 0.260,
            # coverage 0.313, junk 0.319).  A regression to any earlier
            # failure mode still fails loudly.
            "dimensions_ok": float(dim_err.max()) <= 0.20,
            "surface_ok": m["surface_dist_p90"] <= 0.50,
            "coverage_ok": m["coverage_nearest_p95"] <= 0.40,
            "junk_ok": junk_share <= 0.50,
        }
        return m, ok

    verts, _rgb, faces = read_ply_mesh(mesh_path)
    metrics, raw_ok = score(verts)
    metrics.update({"present": True, "faces": int(len(faces)),
                    "unseen_gt_faces": unseen_faces})
    checks = {f"mesh_{key}": value for key, value in raw_ok.items()}

    # Phase 6: the decimated BUSSID-facing mesh, scored the same way plus
    # the triangle budget that keeps the model light.  Absent until the
    # lowpoly stage has run, so sparse-only evaluations stay 4/4.
    low_path = Path(project_dir) / "lowpoly" / "lowpoly.ply"
    if low_path.is_file():
        low_v, _low_rgb, low_f = read_ply_mesh(low_path)
        low_m, low_ok = score(low_v)
        low_m["faces"] = int(len(low_f))
        target = None
        low_meta = Path(project_dir) / "lowpoly" / "lowpoly_meta.json"
        if low_meta.is_file():
            try:
                target = int(json.loads(
                    low_meta.read_text(encoding="utf-8"))["target_triangles"])
            except (OSError, ValueError, KeyError, TypeError):
                target = None
        low_m["target_triangles"] = target
        if target:
            # A hair over budget is fine; never reaching it is not.
            low_ok["triangles_ok"] = len(low_f) <= int(target * 1.1)
        low_m["checks"] = {f"lowpoly_{key}": value
                           for key, value in low_ok.items()}
        metrics["lowpoly"] = low_m
        checks.update(low_m["checks"])

    metrics["checks"] = checks
    return metrics


def evaluate_texture(project_dir: Path) -> Dict[str, Any]:
    """Phase 7 checks: files, atlas coverage, bake chain, cross view.

    Absent until the texture stage has run, so earlier evaluations stay
    unchanged.  Thresholds are calibrated against known-bad variants
    measured with the stage's own check code: a spatially shifted atlas
    (simulating a broken bbox -> pack -> UV chain) scores mapping p50
    58.4 / cross-view p50 60.3, a black atlas ~105, and an atlas that
    was never baked fills ~0.000 of the texture.  Each threshold sits at
    least 1.9x inside those bad values while the measured good run
    (mapping p50 2.2, cross p50 14.3, filled 0.333) passes with
    headroom.  p50 rather than the mean: on a high-frequency texture
    the mean carries a long alignment tail (good mean 8.1 / 34.3) that
    would overlap the bad variants.
    """
    meta_path = Path(project_dir) / "texture" / "texture_meta.json"
    if not meta_path.is_file():
        return {"present": False}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"present": True, "error": str(error),
                "checks": {"texture_files_ok": False}}
    outputs = meta.get("output") or {}
    have_files = bool(outputs) and all(
        (Path(project_dir) / outputs.get(key, "")).is_file()
        for key in ("png", "obj", "mtl"))
    mapping = meta.get("mapping_check") or {}
    cross = meta.get("cross_view_check") or {}
    filled = meta.get("filled_share")
    checks = {
        "texture_files_ok": have_files,
        "texture_coverage_ok":
            filled is not None and filled >= 0.15,
        "texture_mapping_ok":
            mapping.get("samples", 0) >= 30
            and mapping.get("p50_abs_delta_rgb", 999) <= 12.0,
        "texture_cross_view_ok":
            cross.get("samples", 0) >= 30
            and cross.get("p50_abs_delta_rgb", 999) <= 25.0,
    }
    return {
        "present": True,
        "charts": meta.get("charts"),
        "filled_share": filled,
        "mapping_delta_p50": mapping.get("p50_abs_delta_rgb"),
        "cross_view_delta_p50": cross.get("p50_abs_delta_rgb"),
        "cross_view_samples": cross.get("samples"),
        "occluded_skipped": cross.get("occluded_skipped"),
        "view_selection": meta.get("view_selection"),
        "checks": checks,
    }


def evaluate_bussid(project_dir: Path) -> Dict[str, Any]:
    """Phase 9 checks: bussid/ manifest and `.bussidmod` package integrity.

    Absent until the bussid stage has run, so earlier evaluations stay
    unchanged.  These are structural checks, not ground-truth checks -
    what they guard against are the failures that only surface on the
    phone: a truncated archive (interrupted write), a manifest whose
    ``format`` is wrong (import refused), and a mod that carries its
    model but not its material/texture (loads grey in BUSSID - silent at
    exactly the place where nothing can report the problem).
    """
    project_dir = Path(project_dir)
    result_path = project_dir / "exports" / "result.json"
    if not result_path.is_file():
        return {"present": False}
    checks: Dict[str, bool] = {
        "bussid_files_ok": False,
        "bussid_package_ok": False,
        "bussid_textures_ok": False,
    }
    out: Dict[str, Any] = {"present": True}

    # bussid/ prep: the manifest must parse, name the right format, and
    # every asset it declares must exist next to it.
    prep_manifest = project_dir / "bussid" / "manifest.json"
    try:
        prep = json.loads(prep_manifest.read_text(encoding="utf-8"))
        declared = [prep.get("model", ""), *prep.get("assets", [])]
        checks["bussid_files_ok"] = (
            prep.get("format") == "bussid"
            and bool(prep.get("model"))
            and all((project_dir / "bussid" / name).is_file()
                    for name in declared if name)
        )
        out["declared_assets"] = len(prep.get("assets", []))
    except (OSError, ValueError, TypeError) as error:
        out["error"] = str(error)

    # Package: result.json -> file exists -> opens -> valid archive ->
    # root manifest claims bussidmod -> its model and asset list resolve
    # to real entries.
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        package = project_dir / str(result.get("path", ""))
        out["package_bytes"] = result.get("size")
        out["asset_count"] = result.get("asset_count")
        if not package.is_file():
            raise FileNotFoundError(package)
        with zipfile.ZipFile(package) as archive:
            broken = archive.testzip()
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            model = str(manifest.get("model", ""))
            model_in_zip = model in names or f"assets/bussid/{model}" in names
            listed_ok = all(entry in names for entry in
                            manifest.get("assets", []) if entry)
            checks["bussid_package_ok"] = (
                broken is None
                and manifest.get("format") == "bussidmod"
                and bool(model)
                and model_in_zip
                and listed_ok
            )
            checks["bussid_textures_ok"] = (
                any(n.lower().endswith(".mtl") for n in names)
                and any(n.lower().endswith((".png", ".jpg", ".jpeg"))
                        for n in names)
            )
            out["entries"] = len(names)
    except (OSError, ValueError, TypeError, zipfile.BadZipFile,
            KeyError) as error:
        out.setdefault("error", str(error))

    out["checks"] = checks
    return out


def evaluate(project_dir: Path, gt: Dict[str, Any]) -> Dict[str, Any]:
    """Score a completed sparse reconstruction against ground truth."""
    project_dir = Path(project_dir)
    sfm_path = project_dir / "sfm" / "sfm.json"
    if not sfm_path.is_file():
        return {"ok": False, "reason": f"missing {sfm_path}"}
    sfm = json.loads(sfm_path.read_text(encoding="utf-8"))

    gt_by_name = _gt_name_map(project_dir, gt)
    matched: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    # name, est_center, gt_center, est_R
    for image in sfm["images"]:
        entry = gt_by_name.get(image["name"])
        if entry is None:
            continue
        est_R = np.array(image["R"], dtype=np.float64).reshape(3, 3)
        est_t = np.array(image["t"], dtype=np.float64)
        est_center = -est_R.T @ est_t
        gt_R = np.array(entry["R"], dtype=np.float64).reshape(3, 3)
        gt_center = np.array(entry["center"], dtype=np.float64)
        matched.append((image["name"], est_center, gt_center, est_R, gt_R))

    if len(matched) < 3:
        return {"ok": False, "reason": f"only {len(matched)} cameras matched",
                "registered": len(sfm["images"])}

    names = [m[0] for m in matched]
    est_c = np.array([m[1] for m in matched])
    gt_c = np.array([m[2] for m in matched])
    scale, Rm, tvec = umeyama(gt_c, est_c)          # GT -> estimated frame
    aligned_gt = (scale * (gt_c @ Rm.T)) + tvec
    position_errors = np.linalg.norm(aligned_gt - est_c, axis=1)
    scene = float(gt.get("scene_size", 4.2))
    rel_errors = []
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        est_a = next(m for m in matched if m[0] == a)
        est_b = next(m for m in matched if m[0] == b)
        gt_rel = est_b[4] @ est_a[4].T
        est_rel = est_b[3] @ est_a[3].T
        rel_errors.append(rotation_angle_deg(est_rel.T @ gt_rel))

    report = json.loads((project_dir / "reconstruction_report.json")
                        .read_text(encoding="utf-8")) \
        if (project_dir / "reconstruction_report.json").is_file() else {}

    metrics = {
        "ok": True,
        "registered": len(sfm["images"]),
        "total": sfm.get("report", {}).get("total_images", len(gt["images"])),
        "sparse_points": sfm.get("points"),
        "mean_reprojection_error_px": report.get("average_reprojection_error_px"),
        "median_reprojection_error_px": report.get("median_reprojection_error_px"),
        "alignment_scale": round(scale, 4),
        "mean_camera_position_error_m": round(float(position_errors.mean()), 4),
        "camera_position_error_pct_of_scene":
            round(100.0 * float(position_errors.mean()) / scene, 2),
        "max_camera_position_error_m": round(float(position_errors.max()), 4),
        "mean_relative_rotation_error_deg": round(float(np.mean(rel_errors)), 3),
        "max_relative_rotation_error_deg": round(float(np.max(rel_errors)), 3),
        "estimated_focal_px": round(float(sfm["images"][0]["focal"]), 1),
        "ground_truth_focal_px": gt["focal"],
    }
    metrics["checks"] = {
        "registered_ratio_ok":
            metrics["registered"] >= 0.8 * metrics["total"],
        "reprojection_ok":
            (metrics["mean_reprojection_error_px"] or 99) < 3.0,
        "camera_positions_ok":
            metrics["camera_position_error_pct_of_scene"] < 12.0,
        "rotations_ok":
            metrics["mean_relative_rotation_error_deg"] < 4.0,
    }
    mesh = evaluate_mesh(project_dir, gt, (scale, Rm, tvec))
    metrics["mesh"] = mesh
    if mesh.get("checks"):
        metrics["checks"].update(mesh["checks"])
    texture = evaluate_texture(project_dir)
    if texture.get("present"):
        metrics["texture"] = texture
        metrics["checks"].update(texture.get("checks", {}))
    bussid = evaluate_bussid(project_dir)
    if bussid.get("present"):
        metrics["bussid"] = bussid
        metrics["checks"].update(bussid.get("checks", {}))
    metrics["passed"] = all(metrics["checks"].values())
    return metrics


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render", help="render a synthetic dataset")
    p_render.add_argument("out_dir")
    p_render.add_argument("--views", type=int, default=48)
    p_render.add_argument("--fov", type=float, default=65.0,
                          help="horizontal field of view in degrees")
    p_render.add_argument("--focal", type=float, default=None,
                          help="focal in px (default: derived from --fov)")

    p_eval = sub.add_parser("evaluate", help="score an existing project")
    p_eval.add_argument("project")
    p_eval.add_argument("--gt", default=None,
                        help="gt.json path (default: <project>/input/gt.json)")

    p_run = sub.add_parser("run", help="render into a project and run the "
                                       "full pipeline, then evaluate")
    p_run.add_argument("--project", default="synth_selftest")
    p_run.add_argument("--views", type=int, default=48)
    p_run.add_argument("--fov", type=float, default=65.0)
    p_run.add_argument("--focal", type=float, default=None)
    p_run.add_argument("--threads", type=int, default=2)
    p_run.add_argument("--frame-interval", type=float, default=0.25)
    p_run.add_argument("--quality", default="low")

    args = parser.parse_args(argv)

    if args.command == "render":
        payload = render_dataset(Path(args.out_dir), n_views=args.views,
                                 focal=args.focal, fov_degrees=args.fov)
        print(f"rendered {payload['camera_count']} views into {args.out_dir}")
        return 0

    if args.command == "evaluate":
        project = Path(args.project).resolve()
        gt_path = Path(args.gt) if args.gt else project / "input" / "gt.json"
        gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
        metrics = evaluate(project, gt)
        print(json.dumps(metrics, indent=2))
        return 0 if metrics.get("passed") else 1

    # run
    sys.path.insert(0, str(BACKEND_DIR))
    from reconstruction import ReconConfig, run_pipeline

    projects_dir = BACKEND_DIR.parent / "projects"
    project_dir = projects_dir / args.project
    input_dir = project_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    print("rendering synthetic dataset ...")
    render_dataset(input_dir, n_views=args.views, focal=args.focal,
                   fov_degrees=args.fov)
    cfg = ReconConfig.for_project(
        project_dir, quality=args.quality, threads=args.threads,
        frame_interval=args.frame_interval, cpu_only=True)
    cfg.max_frames = args.views * 2
    started = time.time()
    try:
        run_pipeline(cfg)
    except Exception as exc:  # noqa: BLE001 - report then evaluate anyway
        print(f"\npipeline error: {exc}")
    print(f"pipeline wall time: {time.time() - started:.1f}s")
    metrics = evaluate(project_dir,
                       json.loads((input_dir / "gt.json").read_text()))
    print(json.dumps(metrics, indent=2))
    return 0 if metrics.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
