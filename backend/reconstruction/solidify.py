"""Phase 5b: the solidification gate - a watertight deliverable or a failure.

The mesh stage fuses depth maps into a TSDF and extracts its surface, but two
things still leave the shell open afterwards:

* a solid that reaches the array wall has its surface truncated where the grid
  ends (marching cubes only emits cells that fit inside the array), and
* ``trimesh.Trimesh.fill_holes`` only patches short, unambiguous boundary
  loops, so anything wider than a few edges stays open forever.

Neither shows up in the face count.  The file parses, renders and looks
complete while shipping hundreds or thousands of open edges - that is exactly
the "incomplete model" complaint: a hollow shell is not a solid.

This stage closes every remaining boundary loop, keeps a single body, and
**refuses to pass unless the result is watertight**.  A pipeline that cannot
build a solid says so, rather than quietly handing over a hollow one.  It sits
between ``mesh`` and ``lowpoly`` so the decimator, the texture bake and every
export downstream all read the guaranteed solid instead of the raw extract.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import ReconConfig, ReconstructionError
from .ply import read_ply_mesh, write_ply_mesh
from .state import StageContext, StateStore


def _directed_edges(faces: np.ndarray) -> np.ndarray:
    """All directed edges, three per face, in a stable block order.

    Block order (faces' ``(v0,v1)``, then ``(v1,v2)``, then ``(v2,v0)``) is
    what lets a caller recover the owning face of edge ``i`` as ``i % F``.
    """
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                           faces[:, [2, 0]]])


def _edge_audit(faces: np.ndarray, n_vertices: int
                ) -> Tuple[np.ndarray, int, int]:
    """Boundary edges plus the two ways a mesh fails to be watertight.

    A solid needs *every* undirected edge shared by exactly two faces.  Edges
    with one face are holes; edges with three or more are non-manifold, and no
    amount of hole filling repairs those, so they are reported separately
    rather than hidden inside a single number.
    """
    if not len(faces):
        return np.empty((0, 2), dtype=np.int64), 0, 0
    edges = _directed_edges(faces)
    forward = edges[:, 0] * np.int64(n_vertices) + edges[:, 1]
    reverse = edges[:, 1] * np.int64(n_vertices) + edges[:, 0]
    has_opposite = np.isin(reverse, forward)
    boundary = edges[~has_opposite]
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    _uniq, counts = np.unique(lo * np.int64(n_vertices) + hi,
                              return_counts=True)
    open_edges = int((counts == 1).sum())
    non_manifold = int((counts > 2).sum())
    return np.ascontiguousarray(boundary, dtype=np.int64), open_edges, \
        non_manifold


def _chain_loops(boundary: np.ndarray) -> Tuple[List[List[int]], int]:
    """Chain boundary edges into closed loops.

    Returns the loops and the number of edges that could not be chained, which
    only happens where a boundary vertex has more than one outgoing edge - a
    non-manifold boundary.  Those edges cannot be patched by triangulation, so
    they are counted out and left for the gate to reject.
    """
    if not len(boundary):
        return [], 0
    starts = boundary[:, 0]
    if len(np.unique(starts)) != len(starts):
        # Several outgoing edges at some vertex: not a collection of simple
        # cycles, so no triangulation of these edges is well defined.
        return [], int(len(boundary))
    nxt = {int(a): int(b) for a, b in boundary}
    loops: List[List[int]] = []
    seen = set()
    for start in nxt:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur = nxt[start]
        while cur != start:
            # Two ways a walk dies: it returns into a vertex already claimed
            # by another walk, or it reaches a vertex no boundary edge starts
            # from.  The second is an *open chain* - a boundary that trails
            # off instead of closing - and it is common on decimated meshes,
            # so it must end the walk quietly rather than raise KeyError.
            if cur in seen or cur not in nxt:
                loop = None
                break
            seen.add(cur)
            loop.append(cur)
            cur = nxt[cur]
        if loop:
            loops.append(loop)
    unchained = len(boundary) - sum(len(loop) for loop in loops)
    return loops, int(max(0, unchained))


def _fill_loops(loops: List[List[int]]) -> np.ndarray:
    """Triangulate each boundary loop into the faces that close it.

    A loop walked along its boundary edges describes edges the shell *already*
    owns, so the patch has to own their reverses - which means triangulating
    the loop backwards.  A fan needs no geometry library and, unlike
    ``trimesh.fill_holes``, closes loops of any length.
    """
    if not loops:
        return np.empty((0, 3), dtype=np.int64)
    triangles: List[np.ndarray] = []
    for loop in loops:
        if len(loop) < 3:
            continue
        reversed_loop = loop[::-1]
        first = np.int64(reversed_loop[0])
        a = np.asarray(reversed_loop[1:-1], dtype=np.int64)
        b = np.asarray(reversed_loop[2:], dtype=np.int64)
        triangles.append(np.column_stack(
            [np.full(len(a), first, dtype=np.int64), a, b]))
    if not triangles:
        return np.empty((0, 3), dtype=np.int64)
    return np.concatenate(triangles)


def _face_components(faces: np.ndarray, n_vertices: int,
                     *, strict: bool = True) -> Tuple[int, np.ndarray]:
    """Connected components counted by *shared edge*, not shared vertex.

    The mesh stage keeps the largest body using vertex adjacency, which is not
    the notion watertightness uses: two shells that merely touch at a point
    look connected to it but are separate solids (project 47's mesh reported
    one component that way while actually being three).  Faces joined by an
    edge are what makes a body.

    ``strict`` decides how an edge shared by *more* than two faces counts.
    The watertight verdict must ignore those (a 4-face edge is a defect, not a
    connection), but *dropping* bodies must not: faces meeting only across a
    non-manifold edge are still physically attached, and treating them as
    loose debris tears real geometry out of the shell and leaves holes behind.
    So the gate uses ``strict=True`` and keep-largest uses ``strict=False``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_faces = len(faces)
    if not n_faces:
        return 0, np.empty(0, dtype=np.int64)
    edges = _directed_edges(faces)
    face_of_edge = np.tile(np.arange(n_faces, dtype=np.int64), 3)
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    keys = lo * np.int64(n_vertices) + hi
    _uniq, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True)
    shared = counts[inverse] == 2 if strict else counts[inverse] >= 2
    paired = np.flatnonzero(shared)
    if not len(paired):
        return n_faces, np.zeros(n_faces, dtype=np.int64)
    order = np.argsort(inverse[paired], kind="stable")
    paired = paired[order]
    faces_of_edge = face_of_edge[paired]
    group = inverse[paired]
    is_start = np.r_[True, group[1:] != group[:-1]]
    starts = np.flatnonzero(is_start)
    counts_per_group = np.diff(np.r_[starts, len(group)])
    # Star every group on its first face.  Linking only consecutive members
    # would leave a four-face edge split into two pieces that are in fact
    # joined - the whole reason this uses physical connectivity.
    anchor = np.repeat(faces_of_edge[starts], counts_per_group)
    left = anchor[~is_start]
    right = faces_of_edge[~is_start]
    if not len(left):
        return n_faces, np.zeros(n_faces, dtype=np.int64)
    adjacency = coo_matrix((np.ones(len(left), dtype=np.uint8),
                            (left, right)), shape=(n_faces, n_faces))
    n_components, labels = connected_components(adjacency, directed=False)
    return int(n_components), labels


def keep_largest_body(verts: np.ndarray, faces: np.ndarray, rgb: np.ndarray,
                       n_vertices: int
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  Dict[str, Any]]:
    """Drop every body but the largest, by shared-edge connectivity.

    Package-public because ``lowpoly`` runs the same cleanup on what the QEM
    decimator hands back.  Uses physical connectivity (``strict=False``):
    faces meeting across a non-manifold edge are attached to the shell, so
    dropping them would carve holes into a mesh that was closed before the
    drop.
    """
    n_components, labels = _face_components(faces, n_vertices, strict=False)
    stats: Dict[str, Any] = {"components": n_components}
    if n_components <= 1:
        stats.update({"applied": False, "faces_dropped": 0})
        return verts, faces, rgb, stats
    keep = labels == int(np.bincount(labels).argmax())
    stats.update({"applied": True, "faces_dropped": int((~keep).sum())})
    faces = faces[keep]
    used = np.zeros(len(verts), dtype=bool)
    used[faces.reshape(-1)] = True
    if not used.all():
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(int(used.sum()), dtype=np.int64)
        verts = verts[used]
        rgb = rgb[used]
        faces = remap[faces]
    return verts, faces, rgb, stats


def run_solidify(cfg: ReconConfig, store: StateStore,
                 ctx: StageContext) -> Dict[str, Any]:
    """Close every hole, keep one body, and fail unless the shell is solid."""
    src = cfg.d("mesh", "raw", "mesh.ply")
    if not src.is_file():
        raise ReconstructionError(
            "Solidify stage needs the raw mesh (mesh/raw/mesh.ply).",
            suggestion="Run the pipeline through the 'mesh' stage first.",
            details={"project": str(cfg.project_dir)},
        )
    out_dir = cfg.d("mesh", "solid")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mesh_solid.ply"

    ctx.set_command("in-process boundary-loop closure + watertight gate")
    ctx.log(f"solidifying {src.relative_to(cfg.project_dir)}")
    started = time.time()
    ctx.check()

    verts, rgb, faces = read_ply_mesh(src)
    if len(verts) < 4 or len(faces) < 4:
        raise ReconstructionError(
            f"Raw mesh is too small to solidify ({len(verts)} vertices, "
            f"{len(faces)} faces).",
            suggestion="Check mesh/mesh_meta.json (fusion covered_share) - "
                       "a surface this sparse means the depth maps never "
                       "agreed on the object.",
            details={"vertices": len(verts), "faces": len(faces)},
        )
    n_vertices = len(verts)
    _boundary, open_before, non_manifold_before = _edge_audit(faces,
                                                               n_vertices)
    ctx.note(f"solidify input: {len(verts)} vertices, {len(faces)} faces, "
             f"{open_before} open edge(s), "
             f"{non_manifold_before} non-manifold edge(s)")

    # One body first: hole filling is only well defined on the shell we mean
    # to keep, and a pile of detached fragments would each be filled and then
    # all but one thrown away.
    verts, faces, rgb, bodies = keep_largest_body(verts, faces, rgb,
                                                   n_vertices)
    n_vertices = len(verts)
    if bodies.get("applied"):
        ctx.note(f"solidify: kept largest of {bodies['components']} bodies "
                 f"(-{bodies['faces_dropped']} faces)")

    boundary, pre_open, pre_non_manifold = _edge_audit(faces, n_vertices)
    loops, unchained = _chain_loops(boundary)
    patch = _fill_loops(loops)
    faces_added = int(len(patch))
    if faces_added:
        faces = np.concatenate([faces, patch])
    ctx.check()

    # Re-audit from scratch: the patch itself has to be sound, not assumed.
    _boundary_after, final_open, final_non_manifold = _edge_audit(
        faces, n_vertices)
    n_components, _labels = _face_components(faces, n_vertices)

    ctx.note(f"solidify: {len(loops)} hole(s) filled (+{faces_added} faces, "
             f"{pre_open} -> {final_open} open, "
             f"{pre_non_manifold} -> {final_non_manifold} non-manifold); "
             f"{n_components} body/bodies remain")

    report: Dict[str, Any] = {
        "vertices": int(len(verts)),
        "faces": int(len(faces)),
        "watertight": bool(final_open == 0 and final_non_manifold == 0
                           and n_components == 1),
        "open_edges": final_open,
        "non_manifold_edges": final_non_manifold,
        "components": int(n_components),
        "holes_filled": int(len(loops)),
        "faces_added": faces_added,
        "unchained_boundary_edges": int(unchained),
        "bodies": bodies,
        "open_edges_before": open_before,
        "non_manifold_before": non_manifold_before,
        "elapsed_seconds": 0.0,
    }

    report["solid"] = str(out_path.relative_to(cfg.project_dir))
    report["elapsed_seconds"] = round(time.time() - started, 1)
    report["status"] = "passed" if report["watertight"] else "failed"
    # Written on both outcomes: when the gate fails this is the only record of
    # what was actually wrong with the surface, and a failure with nothing to
    # read is just an error message.
    meta_path = out_dir / "solid_meta.json"
    meta_path.write_text(json.dumps(
        {"algorithm": "keep-largest body + boundary-loop fan closure + "
                      "watertight gate",
         **report}, indent=2, default=str), encoding="utf-8")
    ctx.add_output_files([meta_path])

    # The gate.  Everything above is best effort; this is the promise.
    if final_open or final_non_manifold or n_components != 1:
        problems = []
        if final_open:
            problems.append(f"{final_open} open edge(s)")
        if final_non_manifold:
            problems.append(f"{final_non_manifold} non-manifold edge(s)")
        if n_components != 1:
            problems.append(f"{n_components} disconnected bodies")
        raise ReconstructionError(
            "SOLIDIFY FAILED: the mesh is not a watertight solid - "
            + ", ".join(problems) + ".",
            suggestion="Read mesh/solid/solid_meta.json for the exact counts. "
                       "Low fusion.covered_share in mesh/mesh_meta.json, or a "
                       "high solidify.components, means the depth maps covered "
                       "too little of the object to close it: retake the clip "
                       "as one continuous orbit with more overlap between "
                       "frames - a cut or a near-static span leaves regions no "
                       "view ever saw.",
            details=report,
        )

    out = write_ply_mesh(out_path, verts, faces, rgb)
    ctx.add_output_files([out])
    ctx.log(f"watertight solid: {len(verts)} vertices, {len(faces)} faces, "
            f"0 open edges ({report['elapsed_seconds']}s)")
    # Stored explicitly: run_pipeline returns what it is given, but the stage
    # record - what the CLI, the status API and any later diagnosis read -
    # only ever contains what ctx.result() was handed.
    ctx.result(report)
    return report
