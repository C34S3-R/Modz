"""Turn a configured light into something a phone can actually see glow.

The light feature has three stages: detect lamps, store them with a
category and a material name, and deliver a package.  Only the third one
reaches a device - and it was empty.  The delivered MTL held a single
``newmtl vehicle`` with ``illum 1``, the OBJ held a single
``usemtl vehicle``, and the package manifest hardcoded ``"lights": []``.
Detection and the browser preview both worked, so the UI showed lights
that no exported mod could ever emit.

This module closes that gap:

- **assign** - faces inside a light's box (``position``/``rotation``/
  ``scale``, the fields the light editor already edits) are regrouped
  under that light's material, or every face of an ``o``/``g`` group the
  light names.
- **material** - the MTL gains that material with an emissive ``Ke`` and
  ``illum 2``, so both BUSSID (which keys on material *names* such as
  ``LightWhite``/``LuzRoja``) and three.js' MTLLoader (which reads ``Ke``)
  light it up.

Deliberately free of FastAPI/database imports: the reconstruction CLI and
the web pipeline share it, and the CLI must stay importable without a
running server.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "add_light_materials",
    "apply_lights",
    "assign_light_materials",
    "box_transform",
    "emission_for",
    "euler_degrees",
    "load_lights",
    "to_local",
]

# Emission colour per category.  BUSSID names the materials itself
# (LightWhite, LuzRoja, LuzAnaranjada come from light_manager), so colour
# here describes what the lamp *glows* as, not what it is painted.
_EMISSION: Dict[str, Tuple[float, float, float]] = {
    "HEADLIGHTS": (1.0, 1.0, 1.0),
    "REVERSE": (1.0, 1.0, 1.0),
    "FOG": (1.0, 1.0, 1.0),
    "DRL": (1.0, 1.0, 1.0),
    "TAIL_LIGHTS": (1.0, 0.05, 0.05),
    "BRAKE_LIGHTS": (1.0, 0.05, 0.05),
    "INDICATORS": (1.0, 0.45, 0.05),
}
_WHITE = (1.0, 1.0, 1.0)


def emission_for(light: Mapping[str, Any]) -> Tuple[float, float, float]:
    """Glow colour for a light; unknown categories glow white."""
    category = str(light.get("category") or "").upper().replace(" ", "_")
    return _EMISSION.get(category, _WHITE)


def load_lights(project_dir) -> List[Dict[str, Any]]:
    """Lights a project has configured, from wherever the light stage left them.

    Accepts the web pipeline's ``lights/manifest.json`` (``{"lights": [...]}``)
    and a plain ``lights/lights.json`` list, so the reconstruction CLI can
    export lights it did not create.
    """
    import json
    from pathlib import Path

    root = Path(project_dir) / "lights"
    for name in ("manifest.json", "lights.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        items = data.get("lights") if isinstance(data, dict) else data
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _rotation_degrees(degrees: Sequence[float]):
    """Row vector M such that ``local = (world - position) @ M``.

    ``degrees`` is XYZ extrinsic as entered in the light editor (one field
    per axis, step 1).  Returns None when the light is unrotated, which is
    the common case and skips a matrix multiply per face.
    """
    try:
        rx, ry, rz = (float(d) for d in degrees)
    except (TypeError, ValueError):
        return None
    if rx == 0.0 and ry == 0.0 and rz == 0.0:
        return None

    import math

    cx, sx = math.cos(math.radians(rx)), math.sin(math.radians(rx))
    cy, sy = math.cos(math.radians(ry)), math.sin(math.radians(ry))
    cz, sz = math.cos(math.radians(rz)), math.sin(math.radians(rz))
    # R = Rz @ Ry @ Rx  (world = R @ local); we need R^T to go the other way.
    rx_m = ((1, 0, 0), (0, cx, -sx), (0, sx, cx))
    ry_m = ((cy, 0, sy), (0, 1, 0), (-sy, 0, cy))
    rz_m = ((cz, -sz, 0), (sz, cz, 0), (0, 0, 1))

    def mul(a, b):
        return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3))
                           for j in range(3)) for i in range(3))

    rot = mul(mul(rz_m, ry_m), rx_m)
    # Caller reads local_i = sum_k d_k * M[k][i], i.e. local = M^T @ d.
    # We want local = R^T @ d (world = R @ local), so M is R itself.
    return rot


def euler_degrees(matrix: Sequence[Sequence[float]]
                  ) -> Tuple[float, float, float]:
    """XYZ degrees whose ``_rotation_degrees`` reproduces ``matrix``.

    The inverse of ``_rotation_degrees``.  It exists so a caller that has a
    world-space alignment (an Umeyama fit, say) can express it as a light's
    ``rotation`` field instead of assuming the box stays axis-aligned - the
    reconstruction frame is arbitrary, so an unrotated box can land
    anywhere relative to the lamp it is meant to cover.

    ``R = Rz @ Ry @ Rx`` gives ``R[2][0] = -sin(ry)``, ``R[2][1/2]`` in
    ``rx``, ``R[1][0]``/``R[0][0]`` in ``rz``.  At ``ry = +/-90`` those
    couple and ``rz`` becomes unobservable; the well-known fallback pins it
    to zero and reads ``rx`` off the top row.
    """
    import math

    r = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    sy = max(-1.0, min(1.0, -r[2][0]))
    if abs(sy) < 1.0 - 1e-9:
        ry = math.asin(sy)
        rx = math.atan2(r[2][1], r[2][2])
        rz = math.atan2(r[1][0], r[0][0])
    else:
        ry = math.pi / 2.0 if sy > 0 else -math.pi / 2.0
        rx = math.atan2(r[0][1], r[0][2])
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def box_transform(light: Mapping[str, Any]):
    """``(centre, matrix)`` placing the light's box in world space.

    Both ``None`` when the light does not name a usable position.
    """
    try:
        centre = tuple(float(v) for v in light.get("position"))
    except (TypeError, ValueError):
        return None, None
    if len(centre) != 3:
        return None, None
    return centre, _rotation_degrees(light.get("rotation") or (0, 0, 0))


def to_local(point: Sequence[float], centre: Sequence[float],
             matrix: Optional[Sequence[Sequence[float]]]
             ) -> Tuple[float, float, float]:
    """``point`` in the light's own axes, where its box is axis-aligned.

    Shared by the assignment and by anything verifying it, so the two can
    never disagree about which way the box is turned - the rotation has to
    be applied identically on both sides or a face is "outside" a box it
    was assigned from.
    """
    dx = point[0] - centre[0]
    dy = point[1] - centre[1]
    dz = point[2] - centre[2]
    if matrix is None:
        return dx, dy, dz
    return (dx * matrix[0][0] + dy * matrix[1][0] + dz * matrix[2][0],
            dx * matrix[0][1] + dy * matrix[1][1] + dz * matrix[2][1],
            dx * matrix[0][2] + dy * matrix[1][2] + dz * matrix[2][2])


def _half_extents(light: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    """Box half-size, or None when the light does not describe a region.

    The default ``scale`` is ``[1, 1, 1]`` and the default ``position`` is
    the origin, so an untouched light would claim a 2 m cube around the
    model origin and light the entire vehicle.  A light has to be placed
    deliberately before anything is assigned - reporting zero faces is far
    more useful than silently glowing at random.
    """
    try:
        half = tuple(float(v) for v in light.get("scale"))
    except (TypeError, ValueError):
        return None
    if len(half) != 3 or any(v <= 0.0 for v in half):
        return None
    default_box = (half == (1.0, 1.0, 1.0))
    try:
        at_origin = tuple(float(v) for v in light.get("position")) == (0.0, 0.0, 0.0)
    except (TypeError, ValueError):
        at_origin = False
    if default_box and at_origin:
        return None
    return half


def _parse_faces(obj_text: str) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    """(vertices, per-face vertex indices) in file order."""
    verts: List[Tuple[float, float, float]] = []
    faces: List[List[int]] = []
    for line in obj_text.splitlines():
        if line.startswith("v "):
            parts = line.split()
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            ids = []
            for token in line.split()[1:]:
                head = token.split("/", 1)[0]
                if not head:
                    continue
                index = int(head)
                # OBJ allows negative (relative) indices.
                ids.append(len(verts) + index if index < 0 else index - 1)
            if len(ids) >= 3:
                faces.append(ids)
    return verts, faces


def assign_light_materials(obj_text: str,
                           lights: Sequence[Mapping[str, Any]]
                           ) -> Tuple[str, Dict[str, int]]:
    """Regroup OBJ faces under the material of the light that claims them.

    Returns the rewritten OBJ and a ``material -> face count`` tally.  A
    face inside two lights belongs to the first light that names it, which
    keeps the result stable when boxes overlap.
    """
    if not lights:
        return obj_text, {}

    verts, faces = _parse_faces(obj_text)
    if not faces:
        return obj_text, {}

    # Named groups: a light whose object_name matches an o/g section owns
    # every face in it, which is more precise than any bounding box.
    group_of_face: Dict[int, str] = {}
    section: Optional[str] = None
    face_index = 0
    for line in obj_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("o ", "g ")):
            section = stripped.split(None, 1)[1].strip()
        elif stripped.startswith("f "):
            if section:
                group_of_face[face_index] = section
            face_index += 1

    assignment: Dict[int, str] = {}
    claimed = [False] * len(faces)
    counts: Dict[str, int] = {}

    for light in lights:
        material = str(light.get("material") or "").strip()
        if not material:
            continue

        named = str(light.get("object_name") or "").strip()
        wanted = set()
        if named:
            wanted = {i for i, sect in group_of_face.items()
                      if sect == named or sect.lower() == named.lower()}
        if not wanted:
            half = _half_extents(light)
            centre, matrix = box_transform(light)
            if half is None or centre is None:
                continue
            for index, ids in enumerate(faces):
                if claimed[index]:
                    continue
                cx = sum(verts[i][0] for i in ids) / len(ids)
                cy = sum(verts[i][1] for i in ids) / len(ids)
                cz = sum(verts[i][2] for i in ids) / len(ids)
                dx, dy, dz = to_local((cx, cy, cz), centre, matrix)
                if abs(dx) <= half[0] and abs(dy) <= half[1] and abs(dz) <= half[2]:
                    wanted.add(index)
        if not wanted:
            continue

        for index in sorted(wanted):
            if index < len(faces) and not claimed[index]:
                claimed[index] = True
                assignment[index] = material
                counts[material] = counts.get(material, 0) + 1

    if not assignment:
        return obj_text, counts
    return _rewrite_groups(obj_text, assignment), counts


def _rewrite_groups(obj_text: str, assignment: Dict[int, str]) -> str:
    """Emit the OBJ with faces re-run under their assigned material.

    Geometry lines are passed through untouched; only the ``usemtl``
    bookkeeping is regenerated, and it is emitted at material *changes* so
    a single-material model still reads as one contiguous run.
    """
    out: List[str] = []
    pending: List[str] = []
    emitted: Optional[str] = None
    active: Optional[str] = None
    face_index = 0

    def flush() -> None:
        nonlocal pending, emitted
        if pending:
            out.append("usemtl " + (emitted or "vehicle"))
            out.extend(pending)
            pending = []
            emitted = None

    for line in obj_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("usemtl"):
            parts = stripped.split(None, 1)
            active = parts[1].strip() if len(parts) > 1 else active
            continue                      # dropped; re-emitted on flush
        if stripped.startswith("f "):
            want = assignment.get(face_index, active) or active or "vehicle"
            face_index += 1
            if pending and want != emitted:
                flush()
            if not pending:
                emitted = want
            pending.append(line)
            continue
        flush()
        out.append(line)
    flush()

    terminator = "\n" if obj_text.endswith("\n") else ""
    return "\n".join(out) + terminator


def add_light_materials(mtl_text: str,
                        lights: Sequence[Mapping[str, Any]]) -> str:
    """Ensure every light's material exists in the MTL and glows.

    An existing block keeps its ``Kd``/``map_Kd`` (the lamp is painted as
    well as lit); it gains ``Ke`` and ``illum 2``.  A missing block is
    appended whole.  Re-running replaces rather than duplicates.
    """
    materials: Dict[str, Tuple[float, float, float]] = {}
    for light in lights:
        name = str(light.get("material") or "").strip()
        if name and name not in materials:
            materials[name] = emission_for(light)
    if not materials:
        return mtl_text

    lines = mtl_text.splitlines()
    header: List[str] = []
    blocks: List[Tuple[Optional[str], List[str]]] = []
    current: Optional[str] = None
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("newmtl"):
            parts = stripped.split(None, 1)
            current = parts[1].strip() if len(parts) > 1 else ""
            blocks.append((current, []))
        elif current is None:
            header.append(line)
        else:
            blocks[-1][1].append(line)

    by_name: Dict[str, int] = {}
    for index, (name, _body) in enumerate(blocks):
        if name is not None and name not in by_name:
            by_name[name] = index

    for name, colour in materials.items():
        r, g, b = colour
        glow = ["Ka 1.0 1.0 1.0", "Kd %.4f %.4f %.4f" % (r, g, b),
                "Ks 0.0 0.0 0.0", "Ns 0", "illum 2",
                "Ke %.4f %.4f %.4f" % (r, g, b)]
        index = by_name.get(name)
        if index is None:
            blocks.append((name, list(glow)))
            continue
        # Only ever touch the two lines this module owns: a material the
        # project already defined keeps its Kd and map_Kd.  Re-appending
        # the whole glow list instead would duplicate Ka/Kd/Ks/Ns on a
        # second run and make the export non-deterministic.
        body = [ln for ln in blocks[index][1]
                if not ln.strip().lower().startswith(("ke", "illum"))]
        blocks[index] = (name, body + ["illum 2",
                                       "Ke %.4f %.4f %.4f" % (r, g, b)])

    result = list(header)
    for name, body in blocks:
        if name is None:
            result.extend(body)
        else:
            result.append("newmtl " + name)
            result.extend(body)
    return "\n".join(result) + ("\n" if mtl_text.endswith("\n") or result else "")


def apply_lights(obj_text: str, mtl_text: str,
                 lights: Sequence[Mapping[str, Any]]
                 ) -> Tuple[str, str, List[Dict[str, Any]]]:
    """The whole delivery step for one OBJ/MTL pair.

    Returns ``(obj, mtl, applied)`` where ``applied`` is one record per
    light that actually claimed faces - a light that claimed nothing is
    left out and can be reported honestly rather than shipped dark.
    """
    if not lights:
        return obj_text, mtl_text, []

    assigned, counts = assign_light_materials(obj_text, lights)
    mtl_text = add_light_materials(mtl_text, lights)

    applied: List[Dict[str, Any]] = []
    seen: set = set()
    for light in lights:
        name = str(light.get("name") or "")
        material = str(light.get("material") or "").strip()
        if not material or material in seen:
            continue
        seen.add(material)
        applied.append({
            "name": name,
            "category": light.get("category"),
            "material": material,
            "faces": int(counts.get(material, 0)),
            "position": list(light.get("position") or []),
            "scale": list(light.get("scale") or []),
            "glow": list(emission_for(light)),
        })
    return assigned, mtl_text, applied
