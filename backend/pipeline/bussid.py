"""BUSSID asset preparation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from .common import PipelineContext, StageError, referenced_sidecars


def _db_lights(ctx: PipelineContext) -> List[Dict[str, Any]]:
    """Configured lights as plain dicts, vectors decoded from stored JSON.

    The editor writes position/rotation/scale as JSON text, and this is
    the only place the export sees them - nothing downstream of the DB had
    ever read a light back out.
    """
    rows = ctx.db.fetchall(
        "SELECT name, category, position, rotation, scale, material, object_name "
        "FROM lights WHERE project_id = ?",
        (ctx.project_id,),
    )

    def vector(row: Dict[str, Any], key: str, fallback: List[float]):
        raw = row.get(key)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return fallback
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            return list(raw)
        return list(fallback)

    return [{
        "name": row.get("name"),
        "category": row.get("category"),
        "material": row.get("material"),
        "object_name": row.get("object_name"),
        "position": vector(row, "position", [0.0, 0.0, 0.0]),
        "rotation": vector(row, "rotation", [0.0, 0.0, 0.0]),
        "scale": vector(row, "scale", [1.0, 1.0, 1.0]),
    } for row in rows]


def prepare(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.check()
    model = ctx.model_file()
    if not model:
        raise StageError("BUSSID PREPARATION FAILED: no prepared model is available")
    file_row, source = model
    output_dir = ctx.project_dir / "bussid"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source.name
    # Read rather than copy: the light step below rewrites this text, and
    # writing it back only at the end keeps the delivered model differing
    # from the reconstruction by exactly that step.
    obj_text = source.read_text(encoding="utf-8", errors="replace")
    copied_assets = [destination]
    sidecar_suffixes = {".mtl", ".bin", ".png", ".jpg", ".jpeg", ".webp", ".tga", ".bmp"}
    # Sidecars are selected two ways: by stem match (prepared_textured.mtl
    # next to prepared_textured.obj) and by following the model's own
    # mtllib/map_* references (textured.obj keeps referencing textured.mtl
    # after the Blender fallback renamed it) - the latter is what actually
    # decides whether the delivered model shows its texture in game.
    candidates: Dict[str, Path] = {}
    for sibling in source.parent.iterdir():
        if sibling == source or not sibling.is_file() or sibling.suffix.lower() not in sidecar_suffixes:
            continue
        if sibling.stem == source.stem or source.suffix.lower() in {".gltf", ".glb"}:
            candidates[sibling.name] = sibling
    for sidecar in referenced_sidecars(source):
        if sidecar.suffix.lower() in sidecar_suffixes:
            candidates[sidecar.name] = sidecar
    for name, sibling in sorted(candidates.items()):
        sidecar_destination = output_dir / name
        if sibling.resolve() != sidecar_destination.resolve():
            shutil.copy2(sibling, sidecar_destination)
        if sidecar_destination in copied_assets:
            continue
        copied_assets.append(sidecar_destination)
        ctx.register(sidecar_destination, "texture" if sibling.suffix.lower() not in {".mtl", ".bin"} else "metadata")
    # Deliver the lights: regroup faces under each light's material and
    # give that material an emissive Ke, so BUSSID sees a lamp rather than
    # paint and three.js' MTLLoader has something to set emissiveIntensity
    # on in the preview.
    lights = _db_lights(ctx)
    applied: List[Dict[str, Any]] = []
    if lights:
        from lights_apply import add_light_materials, apply_lights

        mtl_assets = [path for path in copied_assets
                      if path.suffix.lower() == ".mtl"]
        if mtl_assets:
            first = mtl_assets[0]
            obj_text, lit_text, applied = apply_lights(
                obj_text,
                first.read_text(encoding="utf-8", errors="replace"),
                lights,
            )
            first.write_text(lit_text, encoding="utf-8")
            for other in mtl_assets[1:]:
                other.write_text(
                    add_light_materials(
                        other.read_text(encoding="utf-8", errors="replace"),
                        lights),
                    encoding="utf-8")
    destination.write_text(obj_text, encoding="utf-8")

    faces_by_material = {item["material"]: item["faces"] for item in applied}
    manifest = {
        "format": "bussid",
        "model": destination.name,
        "assets": [path.name for path in copied_assets],
        "lights": [
            {
                "name": row["name"],
                "material": row["material"],
                "object": row["object_name"],
                "category": row["category"],
                # A light that claimed no faces ships listed but dark;
                # saying so beats reporting a light the mod cannot show.
                "faces": int(faces_by_material.get(row["material"], 0)),
            }
            for row in lights
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ctx.register(destination, "model")
    ctx.register(manifest_path, "metadata")
    ctx.log(
        f"Prepared BUSSID assets from {source.name}",
        stage="BUSSID",
        operation="prepare",
        status="completed",
    )
    return {"model": "bussid/" + destination.name, "manifest": "bussid/manifest.json"}
