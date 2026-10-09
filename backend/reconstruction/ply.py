"""Binary little-endian PLY read/write for xyz + rgb point clouds.

``sfm.export_sparse`` already writes this exact layout for the sparse cloud
and the camera centres; giving the dense and mesh stages one shared reader
and writer stops a second dialect from appearing.  The reader is strict:
it accepts only the property set this module writes and fails loudly on
anything else rather than silently mis-parsing vertex data.

Meshes add a ``face`` element (triangle indices) on top of the same vertex
layout; :func:`write_ply_mesh` / :func:`read_ply_mesh` cover that dialect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)

_PROPERTIES = ("property float x", "property float y", "property float z",
               "property uchar red", "property uchar green", "property uchar blue")


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> Path:
    """Write a binary PLY.  ``xyz`` is (N, 3) float, ``rgb`` is (N, 3) 0-255."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    if rgb.shape != xyz.shape:
        raise ValueError(f"rgb must match xyz, got {rgb.shape} vs {xyz.shape}")

    vertex = np.empty(len(xyz), dtype=_PLY_DTYPE)
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n" + "\n".join(_PROPERTIES) +
        "\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertex.tofile(handle)
    return path


def read_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a PLY written by :func:`write_ply`.  Returns (xyz float32, rgb uint8)."""
    path = Path(path)
    with path.open("rb") as handle:
        count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: PLY header never ended")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("format") and "binary_little_endian" not in text:
                raise ValueError(f"{path}: unsupported PLY format '{text}'")
            if text.startswith("element vertex"):
                count = int(text.split()[-1])
            if text == "end_header":
                break
        if count is None:
            raise ValueError(f"{path}: no 'element vertex' in PLY header")
        data = np.fromfile(handle, dtype=_PLY_DTYPE, count=count)
    if len(data) != count:
        raise ValueError(f"{path}: expected {count} vertices, read {len(data)}")
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


# ---------------------------------------------------------------------------
# Mesh dialect: same vertices + a triangle face element
# ---------------------------------------------------------------------------
# Binary layout per face record: uchar 3, then three little-endian uint32
# indices - the "list uchar uint" convention every mesh PLY reader knows.
_FACE_DTYPE = np.dtype([("count", "u1"), ("i0", "<u4"), ("i1", "<u4"), ("i2", "<u4")])


def write_ply_mesh(path: Path, xyz: np.ndarray, faces: np.ndarray,
                   rgb: Optional[np.ndarray] = None) -> Path:
    """Write a binary triangle-mesh PLY.  ``faces`` is (M, 3) int indices."""
    xyz = np.asarray(xyz, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (M, 3), got {faces.shape}")
    if len(faces) and int(faces.max()) >= len(xyz):
        raise ValueError("face index out of range")
    if rgb is None:
        rgb = np.full(xyz.shape, 128, dtype=np.uint8)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape != xyz.shape:
        raise ValueError(f"rgb must match xyz, got {rgb.shape} vs {xyz.shape}")

    vertex = np.empty(len(xyz), dtype=_PLY_DTYPE)
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    face = np.empty(len(faces), dtype=_FACE_DTYPE)
    face["count"] = 3
    face["i0"], face["i1"], face["i2"] = faces[:, 0], faces[:, 1], faces[:, 2]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n" + "\n".join(_PROPERTIES) +
        f"\nelement face {len(faces)}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertex.tofile(handle)
        if len(face):
            face.tofile(handle)
    return path


def read_ply_mesh(path: Path
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a mesh PLY written by :func:`write_ply_mesh`.

    Returns ``(xyz float32, rgb uint8, faces int64)``.
    """
    path = Path(path)
    with path.open("rb") as handle:
        count = None
        face_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: PLY header never ended")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("format") and "binary_little_endian" not in text:
                raise ValueError(f"{path}: unsupported PLY format '{text}'")
            if text.startswith("element vertex"):
                count = int(text.split()[-1])
            elif text.startswith("element face"):
                face_count = int(text.split()[-1])
            if text == "end_header":
                break
        if count is None or face_count is None:
            raise ValueError(f"{path}: header lacks vertex and face elements")
        vertex = np.fromfile(handle, dtype=_PLY_DTYPE, count=count)
        if len(vertex) != count:
            raise ValueError(f"{path}: expected {count} vertices, "
                             f"read {len(vertex)}")
        face = np.fromfile(handle, dtype=_FACE_DTYPE, count=face_count)
        if len(face) != face_count:
            raise ValueError(f"{path}: expected {face_count} faces, "
                             f"read {len(face)}")
    if face_count and not np.all(face["count"] == 3):
        raise ValueError(f"{path}: only triangle faces are supported")
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1)
    faces = np.stack([face["i0"], face["i1"], face["i2"]], axis=1).astype(np.int64)
    return xyz, rgb, faces
