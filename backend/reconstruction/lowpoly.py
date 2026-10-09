"""Phase 6: light low-poly decimation for BUSSID.

The deliverable is a BUSSID mod, not a film asset: the model has to stay
*light* - a few tens of thousands of triangles, a small file, and a stage
that never burdens the Dell.  The raw marching-cubes surface exists only as
the honest intermediate this stage collapses:

    mesh/raw/mesh.ply  --QEM (fast_simplification)-->  lowpoly/lowpoly.ply

Design decisions:

* **Quadric error metrics with the preset's target.** ``target_triangles``
  comes from the quality preset (low = 20 000, medium = 30 000,
  high = 60 000) and is part of the stage fingerprint, so changing the
  budget re-runs only this stage.  QEM collapses shortest-error edges
  first, which keeps planar vehicle panels crisp at very low counts - the
  right trade for a game model where silhouette matters more than micro
  relief.

* **Colours transfer by nearest source vertex.** The decimator moves
  vertices; frame colours live on the *source* vertices, so every output
  vertex borrows the colour of its closest source counterpart (KD-tree).
  Flat, baked lighting makes nearest-neighbour error invisible at the
  densities involved (mean transfer distance is reported in the meta).

* **Honest passthrough.** A mesh already at or under budget is copied
  through with ``decimated: false`` rather than being "simplified" to the
  same size - the meta says which path ran.

* **Degenerate faces are dropped before QEM.** Marching cubes over a
  noisy TSDF can emit zero-area triangles; their quadric error is zero,
  which would let them dominate collapse decisions.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

import numpy as np

from .config import ReconConfig, ReconstructionError
from .ply import read_ply_mesh, write_ply_mesh
from .state import StageContext, StateStore, memory_pressure

# QEM aggressiveness: the library default.  7 trades a little shape error
# for reliably reaching the target on noisy surfaces; lower would strand
# the mesh above budget, higher starts rounding silhouette corners.
_QEM_AGG = 7.0


def _compact(verts: np.ndarray, faces: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Drop vertices no face references (decimators can strand some)."""
    if not len(faces):
        return verts[:0], faces
    used = np.zeros(len(verts), dtype=bool)
    used[faces.reshape(-1)] = True
    if used.all():
        return verts, faces
    remap = np.full(len(verts), -1, dtype=np.int64)
    kept = np.flatnonzero(used)
    remap[kept] = np.arange(len(kept), dtype=np.int64)
    return verts[used], remap[faces]


def run_lowpoly(cfg: ReconConfig, store: StateStore,
                ctx: StageContext) -> Dict[str, Any]:
    src = cfg.d("mesh", "raw", "mesh.ply")
    if not src.is_file():
        raise ReconstructionError(
            "Low-poly stage needs the raw mesh (mesh/raw/mesh.ply).",
            suggestion="Run the pipeline through the 'mesh' stage first.",
            details={"project": str(cfg.project_dir)},
        )
    out_dir = cfg.d("lowpoly")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lowpoly.ply"
    meta_path = out_dir / "lowpoly_meta.json"

    target = max(1, int(cfg.target_triangles))
    ctx.set_command(
        f"in-process QEM decimation to {target} triangles "
        f"(fast_simplification, agg {_QEM_AGG})")
    ctx.log(f"decimating {src.relative_to(cfg.project_dir)} to "
            f"{target} triangles")
    started = time.time()
    ctx.check()

    verts, rgb, faces = read_ply_mesh(src)
    if len(verts) < 4 or len(faces) < 4:
        raise ReconstructionError(
            f"Raw mesh is too small to decimate ({len(verts)} vertices, "
            f"{len(faces)} faces).",
            suggestion="The mesh stage produced a fragment - inspect "
                       "mesh/mesh_meta.json fusion and prune statistics.",
            details={"vertices": int(len(verts)), "faces": int(len(faces))},
        )
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    # Zero-area triangles: zero quadric error, outsized QEM influence.
    degenerate = ((faces[:, 0] == faces[:, 1])
                  | (faces[:, 1] == faces[:, 2])
                  | (faces[:, 0] == faces[:, 2]))
    removed_degenerate = int(degenerate.sum())
    if removed_degenerate:
        faces = faces[~degenerate]
    ctx.note(f"input: {len(verts)} vertices, {len(faces)} faces"
             + (f" ({removed_degenerate} degenerate dropped)"
                if removed_degenerate else ""))

    decimated = False
    transfer_p95 = 0.0
    if len(faces) > target:
        try:
            import fast_simplification
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ReconstructionError(
                "fast_simplification is required for the low-poly stage.",
                suggestion="python3 -m pip install fast-simplification",
                details={"error": str(exc)},
            ) from exc
        src_tree = None
        try:
            from scipy.spatial import cKDTree
            src_tree = cKDTree(verts.astype(np.float64))
        except ImportError:
            src_tree = None          # colour transfer falls back to index copy

        new_pts, new_tri = fast_simplification.simplify(
            verts.astype(np.float64),
            np.ascontiguousarray(faces, dtype=np.int32),
            target_count=target, agg=_QEM_AGG)
        verts_out = np.ascontiguousarray(new_pts, dtype=np.float32)
        faces_out = np.ascontiguousarray(new_tri, dtype=np.int64)
        verts_out, faces_out = _compact(verts_out, faces_out)

        if src_tree is not None and len(verts_out):
            _dist, nn = src_tree.query(verts_out.astype(np.float64), k=1)
            transfer_p95 = float(np.percentile(_dist, 95))
            rgb_out = rgb[np.asarray(nn, dtype=np.int64)]
        else:
            rgb_out = rgb[:0]
        decimated = True
        ctx.note(f"QEM: {len(verts_out)} vertices, {len(faces_out)} faces "
                 f"(colour transfer p95 {transfer_p95:.4f})")
    else:
        verts_out, faces_out, rgb_out = verts, faces, rgb
        ctx.note(f"already at budget ({len(faces)} <= {target}) - "
                 f"passing through unchanged")

    if len(faces_out) < 4:
        raise ReconstructionError(
            f"Decimation collapsed the mesh to {len(faces_out)} faces.",
            suggestion="Raise RECON_TARGET_TRIANGLES or inspect the raw "
                       "mesh; the input geometry is degenerate.",
            details={"faces": int(len(faces_out))},
        )
    ctx.check()

    out = write_ply_mesh(out_path, verts_out, faces_out, rgb_out)
    ctx.add_output_files([out])

    meta: Dict[str, Any] = {
        "fingerprint": cfg.lowpoly_fingerprint(),
        "algorithm": f"QEM decimation (fast_simplification, agg {_QEM_AGG})",
        "input": {
            "source": str(src.relative_to(cfg.project_dir)),
            "vertices": int(len(verts)),
            "faces": int(len(faces)),
            "degenerate_dropped": removed_degenerate,
        },
        "target_triangles": target,
        "decimated": decimated,
        "vertices": int(len(verts_out)),
        "faces": int(len(faces_out)),
        "reduction": round(1.0 - len(faces_out) / max(1, len(faces)), 4),
        "colour_transfer_p95": round(transfer_p95, 5),
        "output": str(out.relative_to(cfg.project_dir)),
        "elapsed_seconds": round(time.time() - started, 2),
        "memory_peak_gb": memory_pressure(cfg.ram_limit_gb)["process_peak_gb"],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ctx.add_output_files([meta_path])

    ctx.log(
        f"low-poly mesh: {len(faces_out)} faces "
        f"({meta['reduction']:.0%} reduction, target {target}, "
        f"{meta['elapsed_seconds']}s)",
        operation="complete",
    )
    result = {
        "lowpoly": meta["output"],
        "vertices": meta["vertices"],
        "faces": meta["faces"],
        "target_triangles": target,
        "decimated": decimated,
        "elapsed_seconds": meta["elapsed_seconds"],
    }
    ctx.result(result)
    return result
