"""BUSSID asset preparation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

from .common import PipelineContext, StageError, referenced_sidecars


def prepare(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.check()
    model = ctx.model_file()
    if not model:
        raise StageError("BUSSID PREPARATION FAILED: no prepared model is available")
    file_row, source = model
    output_dir = ctx.project_dir / "bussid"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
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
    manifest = {
        "format": "bussid",
        "model": destination.name,
        "assets": [path.name for path in copied_assets],
        "lights": [
            {
                "name": item["name"],
                "material": item["material"],
                "object": item["object_name"],
            }
            for item in ctx.db.fetchall(
                "SELECT name, material, object_name FROM lights WHERE project_id = ?",
                (ctx.project_id,),
            )
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
