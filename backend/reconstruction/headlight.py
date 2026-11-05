"""Isuzu NQR headlight: a low-poly synthetic benchmark that tests *delivery*.

Three jobs, one CLI::

    headlight.py render <dir> [--views N] [--size WxH]
    headlight.py run    [--project NAME] [...]
    headlight.py lights [--project NAME]

``build_headlight()`` is a watertight ~0.9k-triangle lamp assembly: domed
main lens, amber corner indicator, bezel between them, flat back.  It is a
*test dummy* - proportions are approximate, the topology is not: the mesh is
a heightfield solid, so it is closed by construction and has no unpatched
holes for the solidify stage to trip over.

``render_dataset()`` writes frames + ``gt.json`` in exactly the shape
``selftest.py`` uses, so the shared camera-alignment helpers work unchanged.

``light_test()`` is the point of the exercise.  The light feature has three
chances to deliver - detect, configure, *export* - and only the third one
reaches a phone.  So the test runs the real detection, applies a configured
light to the real prepared model, and then reads the exported OBJ/MTL back
to ask the only question that matters: is anything in there that can glow?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Isuzu NQR lamp assembly, metres.  Width is the giveaway: a 340 x 170 mm
# unit with ~100 mm of depth, which is what makes it recognisable in the
# frame rather than "a box with a bump on it".
W, H, D = 0.34, 0.17, 0.10
NY, NZ = 21, 11                      # grid nodes: front + back share these
FRONT_X, BACK_X = D / 2, -D / 2
LENS = dict(cy=-0.075, cz=0.0, ry=0.088, rz=0.066, height=0.045)
INDICATOR = dict(cy=0.108, cz=0.0, ry=0.052, rz=0.052, height=0.030)

# Atlas layout, as (u0, u1, v0, v1) fractions of a square texture.  Every
# panel gets its own region and every region is isotropic - pixels per
# metre matches on both axes.  That is not cosmetic: matching was scored
# against ground truth and the back panel (the only one that had both
# real content and correct aspect) verified at 91-95% while the side walls
# - one shared panel, featureless, and with the top/bottom collapsed onto
# a single texture row - came back at 4-33%, which is what split the match
# graph and stalled SfM at 5 of 16 images.
ATLAS_SIZE = 1280
FRONT_PANEL = (0.000, 0.490, 0.000, 0.246)
BACK_PANEL = (0.000, 0.490, 0.254, 0.500)
RIGHT_PANEL = (0.500, 0.710, 0.000, 0.246)
LEFT_PANEL = (0.500, 0.710, 0.254, 0.500)
TOP_PANEL = (0.000, 0.210, 0.508, 1.000)
BOTTOM_PANEL = (0.220, 0.430, 0.508, 1.000)
PANELS = {
    "front": FRONT_PANEL, "back": BACK_PANEL, "right": RIGHT_PANEL,
    "left": LEFT_PANEL, "top": TOP_PANEL, "bottom": BOTTOM_PANEL,
}
# Walls are parameterised over this x span (back plane to lens peak).
X_SPAN = (FRONT_X + LENS["height"]) - BACK_X

# Where the configured light sits, in ground-truth model coordinates: the
# lens face, generously padded so reconstruction error cannot make the box
# miss the part it is meant to light.
# `pad` widens that box before it is placed in the fitted frame; the
# delivered mesh's face centres landed just outside the unpadded box,
# which is what made the delivery claim nothing and ship dark.
LIGHT_BOX = dict(center=(0.060, -0.075, 0.0), half=(0.062, 0.092, 0.074), pad=1.7)
# Geometry
# ---------------------------------------------------------------------------
def _bump(Y: np.ndarray, Z: np.ndarray, cy: float, cz: float,
          ry: float, rz: float, height: float) -> np.ndarray:
    """Cosine bump over an ellipse, exactly 0 outside it."""
    d = np.sqrt(((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2)
    inside = d < 1.0
    out = np.zeros_like(Y)
    out[inside] = height * 0.5 * (1.0 + np.cos(math.pi * d[inside]))
    return out


def build_headlight() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (verts, faces, tri_uv): a closed low-poly NQR headlight.

    Front and back are the same grid (so the side walls always find their
    partners - this is what makes the solid watertight without a repair
    pass), with the lens and indicator raised out of the front plane.
    """
    ys = np.linspace(-W / 2, W / 2, NY)
    zs = np.linspace(-H / 2, H / 2, NZ)
    Y, Z = np.meshgrid(ys, zs)                       # (NZ, NY)

    lift = _bump(Y, Z, **LENS) + _bump(Y, Z, **INDICATOR)
    lift[0, :], lift[-1, :] = 0.0, 0.0               # flush rim
    lift[:, 0], lift[:, -1] = 0.0, 0.0

    front = np.stack([FRONT_X + lift, Y, Z], axis=-1).reshape(-1, 3)
    back = np.stack([np.full_like(Y, BACK_X), Y, Z],
                    axis=-1).reshape(-1, 3)
    verts = np.concatenate([front, back], axis=0)

    def f(i: int, j: int) -> int:
        return j * NY + i

    def b(i: int, j: int) -> int:
        return NZ * NY + j * NY + i

    faces: List[Tuple[int, int, int]] = []
    for j in range(NZ - 1):
        for i in range(NY - 1):
            faces += [(f(i, j), f(i + 1, j), f(i + 1, j + 1)),
                      (f(i, j), f(i + 1, j + 1), f(i, j + 1))]
            faces += [(b(i, j), b(i, j + 1), b(i + 1, j + 1)),
                      (b(i, j), b(i + 1, j + 1), b(i + 1, j))]

    sides = []
    for i in range(NY - 1):                          # z = -H/2 (normal -Z)
        sides += [(f(i, 0), b(i, 0), b(i + 1, 0)),
                  (f(i, 0), b(i + 1, 0), f(i + 1, 0))]
    for i in range(NY - 1):                          # z = +H/2 (normal +Z)
        j = NZ - 1
        sides += [(f(i, j), f(i + 1, j), b(i + 1, j)),
                  (f(i, j), b(i + 1, j), b(i, j))]
    for j in range(NZ - 1):                          # y = -W/2 (normal -Y)
        sides += [(f(0, j), f(0, j + 1), b(0, j + 1)),
                  (f(0, j), b(0, j + 1), b(0, j))]
    for j in range(NZ - 1):                          # y = +W/2 (normal +Y)
        i = NY - 1
        sides += [(f(i, j), b(i, j), b(i, j + 1)),
                  (f(i, j), b(i, j + 1), f(i, j + 1))]
    faces += sides

    tris = np.array(faces, dtype=np.int64)
    tris = _orient_outward(verts, tris)
    tri_uv = _face_uvs(verts, tris)
    return verts, tris, tri_uv


def _orient_outward(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Flip any triangle whose normal points back into the solid.

    The assembly is convex enough that "away from the centroid" is a safe
    outward test, and being explicit beats trusting the hand-written
    winding: one inverted face is a hole as far as a z-buffer renderer and
    a back-face culler are concerned.
    """
    centre = verts.mean(axis=0)
    tris = tris.copy()
    for index, tri in enumerate(tris):
        a, b, c = verts[tri]
        normal = np.cross(b - a, c - a)
        outward = (a + b + c) / 3.0 - centre
        if float(np.dot(normal, outward)) < 0.0:
            tris[index] = tri[[0, 2, 1]]
    return tris


def _face_uvs(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Per-corner UVs, one isotropic panel per face of the solid.

    Three separate defects lived here, each costing matches on its own:

    * **Aspect.** The atlas is square and the lamp's front face is 2:1, so
      stretching one panel over the whole texture squeezes the vertical
      axis ~3.8x harder than the horizontal one.  SIFT keys on local
      gradients and a 3.8x anisotropic squash smears them out of
      recognition - 75 features a frame instead of hundreds.
    * **One panel per wall.** Left and right shared a single region, making
      the two sides mirror-identical images of each other 180 degrees
      apart; top and bottom were parameterised over z, which is *constant*
      on those walls, so they collapsed onto a single texture row and gave
      the renderer nothing to sample.
    * **Classification by corner position.** The topmost row of a side wall
      sits at the same z as the top face, so y is tested before z or that
      row files itself under the wrong panel.
    """
    pts = verts[tris]                                # (F, 3, 3)
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]

    def all_where(mask: np.ndarray) -> np.ndarray:
        return mask.all(axis=1)

    def flat(panel: Tuple[float, float, float, float],
             u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u0, u1, v0, v1 = panel
        return np.stack([u0 + u * (u1 - u0), v0 + v * (v1 - v0)], axis=-1)

    ny = (y + W / 2) / W
    nz = (z + H / 2) / H
    nx = (x - BACK_X) / X_SPAN

    is_front = all_where(x >= FRONT_X - 1e-9)
    is_back = all_where(x <= BACK_X + 1e-9)
    wall = ~(is_front | is_back)

    is_right = wall & all_where(y >= W / 2 - 1e-9)
    is_left = wall & all_where(y <= -W / 2 + 1e-9)
    rest = wall & ~(is_right | is_left)
    is_top = rest & all_where(z >= H / 2 - 1e-9)
    is_bottom = rest & all_where(z <= -H / 2 + 1e-9)
    is_top |= rest & ~(is_top | is_bottom)          # safety net

    tri_uv = flat(FRONT_PANEL, ny, nz)
    tri_uv[is_back] = flat(BACK_PANEL, ny, nz)[is_back]
    tri_uv[is_right] = flat(RIGHT_PANEL, nx, nz)[is_right]
    tri_uv[is_left] = flat(LEFT_PANEL, nx, nz)[is_left]
    tri_uv[is_top] = flat(TOP_PANEL, nx, ny)[is_top]
    tri_uv[is_bottom] = flat(BOTTOM_PANEL, nx, ny)[is_bottom]
    return tri_uv


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------
def _industrial(rng: np.random.Generator, h: int, w: int, base,
                rib_step: int) -> np.ndarray:
    """Moulded plastic: ribbing, bosses, lettering runs, scuffs, a landmark.

    Every mark's position and size comes from a seeded generator, so the
    layout is irregular - irregularity is what stops every descriptor
    matching every other one - while staying reproducible between runs.
    The back panel already proved this recipe works: it verified at 91-95%
    where a flat, banded wall managed 4%.
    """
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = base
    yy, xx = np.mgrid[0:h, 0:w]
    if rib_step:
        img[(yy % rib_step) < max(rib_step // 3, 4)] = \
            tuple(min(c + 24, 255) for c in base)

    area = h * w
    for _ in range(max(9, area // 20000)):                 # bolt bosses
        cy = int(rng.integers(34, max(h - 34, 35)))
        cx = int(rng.integers(34, max(w - 34, 35)))
        rr = int(rng.integers(17, 42))
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr * rr
        img[disc] = (106, 110, 116)
        head = ((xx - cx) ** 2 + (yy - cy) ** 2) <= max(rr // 3, 5) ** 2
        img[head] = (158, 162, 168)

    # Density matters as much as distinctiveness: pairs at the awkward
    # face-transition angles were coming back with 16-34 raw matches, and
    # 25 of them have to survive RANSAC before a pair is worth anything.
    # Heights vary too, so the marks land at more than one scale.
    for _ in range(max(90, area // 1700)):
        x0 = int(rng.integers(4, max(w - 60, 5)))
        y0 = int(rng.integers(4, max(h - 26, 5)))
        run = int(rng.integers(14, 58))
        tall = int(rng.integers(7, 16))
        img[y0:y0 + tall, x0:x0 + run] = (186, 190, 196)

    for size in (70, 44, 24):                               # scuffs, 3 scales
        for _ in range(max(4, area // (size * size * 300))):
            x0 = int(rng.integers(0, max(w - size, 1)))
            y0 = int(rng.integers(0, max(h - size, 1)))
            img[y0:y0 + size, x0:x0 + size] = np.clip(
                img[y0:y0 + size, x0:x0 + size].astype(np.int16)
                + rng.integers(-72, 73, size=3), 0, 255).astype(np.uint8)

    # One large landmark, a different shape on every panel, so no other
    # panel can offer a competitor for it - that is what lets a matcher
    # pin a view down across a wide baseline instead of only up close.
    kind = int(rng.integers(0, 3))
    cy = int(rng.integers(h // 4, max(h - h // 4, h // 4 + 1)))
    cx = int(rng.integers(w // 4, max(w - w // 4, w // 4 + 1)))
    rad = int(min(h, w) * 0.32)
    if kind == 0:
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        img[(r2 <= rad * rad) & (r2 >= (rad * 0.70) ** 2)] = (242, 242, 240)
    elif kind == 1:
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad] = (242, 242, 240)
    else:
        half_h = max(rad // 4, 6)
        img[max(cy - half_h, 0):cy + half_h,
            max(cx - rad, 0):min(cx + rad, w)] = (242, 242, 240)
    return img


def make_atlas(size: int = ATLAS_SIZE) -> np.ndarray:
    """A square atlas with one isotropic panel per face of the solid.

    Content is sized for what survives the trip to the frame: a panel line
    52 atlas pixels wide lands at ~22 frame pixels, lens fluting at ~14,
    lettering runs at ~8 - three scales, which is what SIFT keys on.  Flat
    or periodic colour is the one input that stalls matching, so the ribs
    and hatching that remain are all overlaid with irregular marks.
    """
    rng = np.random.default_rng(11)
    S = size
    img = np.zeros((S, S, 3), np.uint8)

    # Bold blocks everywhere first, so the panel margins are still
    # structured if a face ever lands off its panel.
    img[:] = (30, 32, 36)
    for _ in range(600):
        x0 = int(rng.integers(0, S - 44))
        y0 = int(rng.integers(0, S - 44))
        img[y0:y0 + 32, x0:x0 + 32] = rng.integers(0, 255, size=3,
                                                    dtype=np.uint8)

    def panel_px(panel):
        u0, u1, v0, v1 = panel
        return int(round(v0 * S)), int(round(v1 * S)), \
            int(round(u0 * S)), int(round(u1 * S))

    def panel_size(panel):
        r0, r1, c0, c1 = panel_px(panel)
        return r1 - r0, c1 - c0

    def stamp(panel, drawn):
        r0, r1, c0, c1 = panel_px(panel)
        img[r0:r1, c0:c1] = drawn

    # --- front: bezel, domed lens, corner indicator ------------------------
    r0, r1, c0, c1 = panel_px(FRONT_PANEL)
    fh, fw = r1 - r0, c1 - c0
    fy, fx = np.mgrid[0:fh, 0:fw]
    front = np.zeros((fh, fw, 3), np.uint8)
    front[:] = (44, 47, 52)
    # Irregular panel lines: a uniform grid is a field of identical
    # corners, and identical corners are what the ratio test cannot tell
    # apart from one another.
    for xpos in rng.choice(np.arange(24, max(fw - 24, 25)), size=7,
                           replace=False):
        front[:, int(xpos):int(xpos) + 5] = (78, 82, 88)
    for ypos in rng.choice(np.arange(20, max(fh - 20, 21)), size=4,
                           replace=False):
        front[int(ypos):int(ypos) + 5, :] = (78, 82, 88)
    for _ in range(18):                                  # coarse scuffs
        sz = int(rng.integers(26, 74))
        x0 = int(rng.integers(0, max(fw - sz, 1)))
        y0 = int(rng.integers(0, max(fh - sz, 1)))
        front[y0:y0 + sz, x0:x0 + sz] = np.clip(
            front[y0:y0 + sz, x0:x0 + sz].astype(np.int16)
            + rng.integers(-58, 59, size=3), 0, 255).astype(np.uint8)
    # Bezel lettering and bosses, drawn before the lens so anything the
    # lamp covers is simply overwritten - which is how a real assembly
    # reads anyway, and the bezel was the emptiest thing in the frame.
    for _ in range(90):
        x0 = int(rng.integers(4, max(fw - 58, 5)))
        y0 = int(rng.integers(4, max(fh - 26, 5)))
        run = int(rng.integers(14, 54))
        tall = int(rng.integers(7, 15))
        front[y0:y0 + tall, x0:x0 + run] = (186, 190, 196)
    for _ in range(7):
        bx0 = int(rng.integers(34, max(fw - 34, 35)))
        by0 = int(rng.integers(30, max(fh - 30, 31)))
        rr = int(rng.integers(14, 30))
        disc = ((fx - bx0) ** 2 + (fy - by0) ** 2) <= rr * rr
        front[disc] = (106, 110, 116)

    lc = ((LENS["cy"] + W / 2) / W) * fw
    lr = (LENS["ry"] / W) * fw
    lz = ((LENS["cz"] + H / 2) / H) * fh
    lh = (LENS["rz"] / H) * fh
    d = ((fx - lc) / lr) ** 2 + ((fy - lz) / lh) ** 2
    lens = d <= 1.0
    # Flute spacing drifts across the lens and its brightness is enveloped
    # top to bottom: a perfectly periodic comb hands SIFT the same
    # descriptor at every flute and the ratio test rejects all of them.
    phase = (fx - lc) * (1.0 / 16.0 + 0.9 * (fx - lc) / (lr * lr))
    flute = 0.5 + 0.5 * np.cos(phase * 2 * np.pi)
    flute *= 0.70 + 0.30 * (0.5 + 0.5 * np.cos((fy - lz) / 47.0 * 2 * np.pi))
    base = 158 + 86 * flute
    lens_rgb = np.stack([base, base, np.minimum(base + 16, 255)],
                        axis=-1).astype(np.uint8)
    front[lens] = lens_rgb[lens]
    front[lens & (d > 0.86)] = (252, 252, 250)
    for frac in (0.31, 0.57, 0.79):                       # segment breaks
        bx = int(round(lc + (frac - 0.5) * 2.0 * lr))
        front[(np.abs(fx - bx) <= 4) & lens] = (34, 36, 40)

    ic = ((INDICATOR["cy"] + W / 2) / W) * fw
    ir = (INDICATOR["ry"] / W) * fw
    iz = ((INDICATOR["cz"] + H / 2) / H) * fh
    ih = (INDICATOR["rz"] / H) * fh
    di = ((fx - ic) / ir) ** 2 + ((fy - iz) / ih) ** 2
    lamp = di <= 1.0
    # Coprime x/y periods: the tile is 17*23 px across, wider than the
    # panel, so the pattern never repeats within it.
    cross = ((fx // 17 + fy // 23) % 2).astype(bool)
    amber = np.where(cross[..., None], np.array([28, 150, 246]),
                     np.array([72, 192, 252])).astype(np.uint8)
    front[lamp] = amber[lamp]
    front[lamp & (di > 0.82)] = (184, 234, 255)
    for frac in (0.36, 0.68):                             # prism dividers
        bx = int(round(ic + (frac - 0.5) * 2.0 * ir))
        front[(np.abs(fx - bx) <= 5) & lamp] = (26, 60, 96)
    stamp(FRONT_PANEL, front)

    # --- every other face gets its own irregular moulding ------------------
    stamp(BACK_PANEL, _industrial(rng, *panel_size(BACK_PANEL),
                                  (38, 40, 44), 30))
    stamp(RIGHT_PANEL, _industrial(rng, *panel_size(RIGHT_PANEL),
                                   (40, 43, 48), 34))
    stamp(LEFT_PANEL, _industrial(rng, *panel_size(LEFT_PANEL),
                                  (44, 41, 38), 26))
    stamp(TOP_PANEL, _industrial(rng, *panel_size(TOP_PANEL),
                                 (36, 39, 43), 38))
    stamp(BOTTOM_PANEL, _industrial(rng, *panel_size(BOTTOM_PANEL),
                                    (46, 44, 40), 30))

    grain = rng.integers(-6, 7, size=(S, S, 3))
    return np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    from .selftest import look_at
    return look_at(np.asarray(eye, dtype=np.float64),
                   np.asarray(target, dtype=np.float64), up)


def render_dataset(out_dir: Path, n_views: int = 16, width: int = 400,
                   height: int = 300, fov_degrees: float = 65.0,
                   seed: int = 5) -> Dict[str, Any]:
    """Orbit the lamp and write frames + gt.json (selftest's shape)."""
    import cv2
    from .selftest import render_view

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    rng = np.random.default_rng(seed)
    verts, tris, tri_uv = build_headlight()
    atlas = make_atlas()
    focal = (width / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    K = np.array([[focal, 0.0, width / 2.0],
                  [0.0, focal, height / 2.0],
                  [0.0, 0.0, 1.0]])
    target = np.array([0.02, 0.0, 0.0])
    # Tight enough that the lamp fills ~2/3 of the frame width: at 0.55 m
    # it projected to 200 px, which starved matching of features.
    radius = 0.42

    entries = []
    for index in range(n_views):
        theta = 2 * math.pi * index / n_views + float(rng.uniform(-0.04, 0.04))
        eye = target + np.array([
            radius * math.cos(theta),
            radius * math.sin(theta),
            0.04 + 0.10 * (0.5 + 0.5 * math.sin(theta * 2.0))
            + float(rng.uniform(-0.03, 0.03)),
        ])
        R, t = _look_at(eye, target)
        image = render_view(verts, tris, tri_uv, [atlas], R, t, K,
                            width, height, rng)
        name = f"headlight_{index + 1:04d}.jpg"
        ok, buf = cv2.imencode(".jpg", image,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"could not encode {name}")
        buf.tofile(str(out_dir / name))
        entries.append({
            "file": name,
            "R": [float(x) for x in R.ravel()],
            "t": [float(x) for x in t.ravel()],
            "center": [float(x) for x in eye],
        })

    gt = {
        "width": width, "height": height, "focal": float(focal),
        "principal_point": [width / 2.0, height / 2.0],
        "camera_count": len(entries),
        "scene_size": W,
        "model": "isuzu-nqr-headlight",
        "images": entries,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "gt.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    return gt


# ---------------------------------------------------------------------------
# Ground truth -> reconstructed frame
# ---------------------------------------------------------------------------
def align_to_gt(project_dir: Path, gt: Dict[str, Any]):
    """Similarity taking GT model coordinates into the reconstructed frame.

    Reuses the selftest's own camera matching + Umeyama fit, so the light
    box is placed by the same alignment that scores geometry - not by a
    guess about which way the reconstruction happens to face.
    """
    from .selftest import _gt_name_map, umeyama

    sfm_path = project_dir / "sfm" / "sfm.json"
    if not sfm_path.is_file():
        return None, "no sfm.json (sparse stage did not complete)"
    sfm = json.loads(sfm_path.read_text(encoding="utf-8"))
    gt_by_name = _gt_name_map(project_dir, gt)

    est_c, gt_c = [], []
    for image in sfm.get("images", []):
        entry = gt_by_name.get(image["name"])
        if entry is None:
            continue
        R = np.array(image["R"], dtype=np.float64).reshape(3, 3)
        t = np.array(image["t"], dtype=np.float64)
        est_c.append(-R.T @ t)
        gt_c.append(np.array(entry["center"], dtype=np.float64))
    if len(est_c) < 3:
        return None, f"only {len(est_c)} cameras matched to ground truth"

    scale, Rm, tvec = umeyama(np.array(gt_c), np.array(est_c))
    return (float(scale), Rm, tvec), f"{len(est_c)} cameras aligned"


def gt_to_est(point, alignment) -> np.ndarray:
    scale, Rm, tvec = alignment
    return scale * (np.asarray(point, dtype=np.float64) @ Rm.T) + tvec


# ---------------------------------------------------------------------------
# The light test
# ---------------------------------------------------------------------------
def light_test(project_dir: Path, gt: Dict[str, Any],
               cfg: Any = None) -> Dict[str, Any]:
    """Does a configured light survive all the way to the exported files?

    Three gates, in the order the feature really runs:

      1. detect    - the real regex over the project's inputs
      2. configure - write the light where the pipeline's light stage
                      leaves it (``lights/manifest.json``), positioned over
                      the lens by ground-truth alignment
      3. deliver   - run the *actual* BUSSID stage, then read back the
                      files a phone would receive, package included

    Gate 3 re-runs the stage rather than inspecting a hand-made file,
    because the defect this test exists for was precisely that the stage
    wrote nothing: testing a helper in isolation would have passed while
    the export stayed dark.
    """
    project_dir = Path(project_dir).resolve()
    checks: Dict[str, bool] = {}
    result: Dict[str, Any] = {"ok": False}

    model_path = _prepared_model(project_dir)
    if model_path is None:
        result["reason"] = "no prepared model (blender stage did not run)"
        result["gates_all"] = {"prepared_model": False}
        return result

    # -- gate 1: detection --------------------------------------------------
    inputs = [{"path": str(p)} for p in
              sorted((project_dir / "input").glob("*.jpg"))[:3]]
    if not inputs:
        inputs = [{"path": str(project_dir / "input" / "headlight_0001.jpg")}]
    material = "LightWhite"
    try:
        sys.path.insert(0, str(BACKEND_DIR))
        from light_manager import _MATERIALS, LightManager

        analysis = LightManager(None, None).analysis(
            {"inputs": inputs}, model_path)
        result["detection"] = {
            "lights": analysis["lights"],
            "warnings": analysis["warnings"],
            "materials": analysis["materials"],
        }
        checks["detect_headlights"] = "HEADLIGHTS" in analysis["lights"]
        material = analysis["materials"].get("HEADLIGHTS") or material
    except Exception as exc:                                   # noqa: BLE001
        result["detection_error"] = f"{type(exc).__name__}: {exc}"
        checks["detect_headlights"] = False

    # -- gate 2: configure a light over the lens ----------------------------
    alignment, why = align_to_gt(project_dir, gt)
    result["alignment"] = why
    if alignment is None:
        checks["configure"] = False
        result["gates"] = checks
        return result

    sys.path.insert(0, str(BACKEND_DIR))
    from lights_apply import euler_degrees

    scale, Rm, _tvec = alignment
    centre = gt_to_est(LIGHT_BOX["center"], alignment)
    # Half-extents scale with the fit, and the box rotates with it too: the
    # reconstruction's axes are arbitrary, so an axis-aligned box in that
    # frame is aligned with nothing in particular.  Expressing the fit's
    # rotation as the light's own `rotation` field also means the delivery
    # step is exercised exactly as the light editor would drive it.
    half = (np.array(LIGHT_BOX["half"], dtype=float) * scale
            * float(LIGHT_BOX.get("pad") or 1.0))
    light = {
        "name": "headlight_lens",
        "category": "HEADLIGHTS",
        "material": material,
        "position": [float(x) for x in centre],
        "rotation": [float(d) for d in euler_degrees(Rm)],
        "scale": [float(x) for x in half],
        "object_name": "",
        "status": "Configured",
    }
    result["light"] = light

    lights_dir = project_dir / "lights"
    lights_dir.mkdir(parents=True, exist_ok=True)
    (lights_dir / "manifest.json").write_text(
        json.dumps({"lights": [light]}, indent=2), encoding="utf-8")
    checks["configure"] = True

    # -- gate 3: deliver ----------------------------------------------------
    if cfg is not None:
        try:
            _run_bussid_stage(cfg)
        except Exception as exc:                               # noqa: BLE001
            result["delivery_error"] = f"{type(exc).__name__}: {exc}"

    try:
        delivered = _measure_delivery(project_dir, light)
    except Exception as exc:                                   # noqa: BLE001
        result["delivery_error"] = f"{type(exc).__name__}: {exc}"
        delivered = {}
    result["delivery"] = delivered

    checks["export_material"] = bool(delivered.get("material_in_mtl"))
    checks["export_emissive"] = bool(delivered.get("emissive_in_mtl"))
    checks["export_faces_lit"] = (delivered.get("lit_faces") or 0) > 0
    checks["export_partial"] = (delivered.get("unlit_faces") or 0) > 0
    checks["export_in_box"] = (delivered.get("outside_box") or 0) == 0
    checks["export_manifest"] = bool(delivered.get("manifest_lists_light"))
    checks["export_package"] = bool(delivered.get("package_material"))

    result["gates"] = checks
    result["gates_all"] = checks
    result["ok"] = all(checks.values())
    return result


def _run_bussid_stage(cfg: Any) -> None:
    """Run the real BUSSID stage through its own store context."""
    sys.path.insert(0, str(BACKEND_DIR))
    import reconstruction
    from reconstruction.state import StateStore

    store = StateStore(cfg.state_path)
    handlers = reconstruction._handlers()
    with store.stage("bussid") as ctx:
        handlers["bussid"](cfg, store, ctx)


def _measure_delivery(project_dir: Path, light: Dict[str, Any]
                      ) -> Dict[str, Any]:
    """Score the files the BUSSID stage actually produced."""
    import zipfile

    sys.path.insert(0, str(BACKEND_DIR))
    import lights_apply

    bussid = project_dir / "bussid"
    material = light["material"]
    box_half = np.array(light["scale"], dtype=float)
    box_centre, box_matrix = lights_apply.box_transform(light)

    out: Dict[str, Any] = {
        "material_in_mtl": False, "emissive_in_mtl": False,
        "manifest_lists_light": False, "lit_faces": 0, "unlit_faces": 0,
        "outside_box": 0, "total_faces": 0, "package_material": False,
    }

    manifest_path = bussid / "manifest.json"
    objs = sorted(bussid.glob("*.obj")) if bussid.is_dir() else []
    mtls = sorted(bussid.glob("*.mtl")) if bussid.is_dir() else []
    if not (manifest_path.is_file() and objs and mtls):
        out["error"] = "bussid/ is missing its model, materials or manifest"
        return out

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("lights") or []:
        if str(item.get("material", "")) == material:
            out["manifest_lists_light"] = True

    mtl_text = mtls[0].read_text(encoding="utf-8", errors="replace")
    obj_text = objs[0].read_text(encoding="utf-8", errors="replace")
    out["material_in_mtl"] = f"newmtl {material}" in mtl_text
    out["emissive_in_mtl"] = _has_emission(mtl_text, material)

    current, verts = None, []
    for line in obj_text.splitlines():
        if line.startswith("v "):
            verts.append(np.array([float(x) for x in line.split()[1:4]]))
        elif line.startswith("usemtl"):
            current = line.split(None, 1)[1].strip()
        elif line.startswith("f "):
            out["total_faces"] += 1
            if current == material:
                out["lit_faces"] += 1
                ids = [int(tok.split("/")[0]) for tok in line.split()[1:4]]
                centroid = np.mean([verts[i - 1] for i in ids], axis=0)
                # Same transform the assignment used - a check written
                # separately would happily disagree about where the box is.
                local = np.array(lights_apply.to_local(
                    centroid, box_centre, box_matrix), dtype=float)
                if np.any(np.abs(local) > box_half * 1.05):
                    out["outside_box"] += 1
            else:
                out["unlit_faces"] += 1

    # The package is the deliverable: a correct bussid/ that never reaches
    # the zip is exactly the failure this gate exists to catch.
    result_path = project_dir / "exports" / "result.json"
    if result_path.is_file():
        try:
            package = project_dir / json.loads(
                result_path.read_text(encoding="utf-8"))["path"]
            with zipfile.ZipFile(package) as archive:
                for name in archive.namelist():
                    if name.lower().endswith(".mtl"):
                        text = archive.read(name).decode("utf-8", "replace")
                        if f"newmtl {material}" in text:
                            out["package_material"] = True
                        break
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            out["package_error"] = str(exc)
    return out


def _prepared_model(project_dir: Path) -> Optional[Path]:
    meta = project_dir / "blender" / "blender_meta.json"
    if meta.is_file():
        try:
            recorded = json.loads(meta.read_text(encoding="utf-8")).get("output")
            if recorded:
                candidate = project_dir / recorded
                if candidate.is_file():
                    return candidate
        except (OSError, ValueError, TypeError):
            pass
    candidates = sorted((project_dir / "blender").glob("prepared_*.obj")) \
        if (project_dir / "blender").is_dir() else []
    return candidates[0] if len(candidates) == 1 else None


def _has_emission(mtl_text: str, material: str) -> bool:
    """True when that material's block carries an emissive (Ke) statement."""
    block, active = [], False
    for line in mtl_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("newmtl"):
            parts = stripped.split(None, 1)
            active = len(parts) > 1 and parts[1].strip() == material
            continue
        if active:
            block.append(stripped.lower())
    return any(b.startswith("ke ") or b == "ke" for b in block)


def _parse_size(text: str) -> Tuple[int, int]:
    width, _, height = text.partition("x")
    return int(width), int(height or width)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render", help="render a synthetic headlight orbit")
    p_render.add_argument("out_dir")
    p_render.add_argument("--views", type=int, default=16)
    p_render.add_argument("--size", default="400x300")

    p_run = sub.add_parser("run", help="render + full pipeline + light test")
    p_run.add_argument("--project", default="isuzu_nqr_headlight")
    p_run.add_argument("--views", type=int, default=16)
    p_run.add_argument("--size", default="400x300")
    p_run.add_argument("--quality", default="low")
    p_run.add_argument("--threads", type=int, default=None)
    p_run.add_argument("--target-triangles", type=int, default=3000)

    p_lights = sub.add_parser("lights", help="score the light feature only")
    p_lights.add_argument("--project", default="isuzu_nqr_headlight")
    p_lights.add_argument("--quality", default="low")
    p_lights.add_argument("--threads", type=int, default=None)
    p_lights.add_argument("--target-triangles", type=int, default=3000)

    args = parser.parse_args(argv)

    if args.command == "render":
        w, h = _parse_size(args.size)
        payload = render_dataset(Path(args.out_dir), n_views=args.views,
                                 width=w, height=h)
        print(f"rendered {payload['camera_count']} views into {args.out_dir}")
        return 0

    sys.path.insert(0, str(BACKEND_DIR))
    from reconstruction import ReconConfig, run_pipeline

    project_dir = BACKEND_DIR.parent / "projects" / args.project

    if args.command == "lights":
        gt = json.loads((project_dir / "input" / "gt.json")
                        .read_text(encoding="utf-8"))
        cfg = ReconConfig.for_project(project_dir, quality=args.quality,
                                      threads=args.threads,
                                      target_triangles=args.target_triangles,
                                      cpu_only=True)
        report = light_test(project_dir, gt, cfg)
        print(json.dumps(report, indent=2))
        print(f"\nLIGHT TEST: {'PASS' if report.get('ok') else 'FAIL'}")
        return 0 if report.get("ok") else 1

    # run
    w, h = _parse_size(args.size)
    input_dir = project_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    print("rendering headlight orbit ...")
    gt = render_dataset(input_dir, n_views=args.views, width=w, height=h)

    cfg = ReconConfig.for_project(
        project_dir, quality=args.quality, threads=args.threads,
        target_triangles=args.target_triangles, cpu_only=True)
    cfg.max_frames = args.views * 2

    started = time.time()
    try:
        run_pipeline(cfg)
    except Exception as exc:                                   # noqa: BLE001
        print(f"\npipeline error: {exc}")
    wall = time.time() - started
    print(f"pipeline wall time: {wall:.1f}s")

    report = light_test(project_dir, gt, cfg)
    print(json.dumps(report, indent=2))
    print(f"\nLIGHT TEST: {'PASS' if report.get('ok') else 'FAIL'}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
