"""Validate generated reconstruction artifacts before downstream processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .common import PipelineContext, StageError


_MODEL_EXTENSIONS = {".obj", ".fbx", ".glb", ".gltf", ".blend", ".stl", ".ply"}


def validate(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.check()
    candidates = ctx.files(
        roots=["reconstruction"],
        extensions=_MODEL_EXTENSIONS,
    )
    if not candidates:
        candidates = ctx.files(roots=["input"], kinds=["model"])
    if not candidates:
        raise StageError("MESH VALIDATION FAILED: no reconstructed or input model was found")
    file_row, path = candidates[0]
    if not path.is_file() or path.stat().st_size == 0:
        raise StageError("MESH VALIDATION FAILED: model file is empty")
    vertices = 0
    faces = 0
    materials = 0
    missing_materials: list[str] = []
    if path.suffix.lower() == ".obj":
        try:
            with path.open("r", errors="replace") as handle:
                for line in handle:
                    if line.startswith("v "):
                        vertices += 1
                    elif line.startswith("f "):
                        faces += 1
                    elif line.startswith("mtllib "):
                        materials += 1
                        material_name = line.split(maxsplit=1)[1].strip()
                        if material_name:
                            material_path = (path.parent / material_name).resolve()
                            try:
                                material_path.relative_to(path.parent.resolve())
                            except ValueError:
                                missing_materials.append(material_name)
                            else:
                                if not material_path.is_file():
                                    missing_materials.append(material_name)
        except OSError as exc:
            raise StageError("MESH VALIDATION FAILED: model could not be read") from exc
        if vertices == 0 or faces == 0:
            raise StageError(
                "MESH VALIDATION FAILED: OBJ contains no usable vertices or faces",
                details={"vertices": vertices, "faces": faces},
            )
        if missing_materials:
            raise StageError(
                "MESH VALIDATION FAILED: OBJ references missing material files",
                details={"materials": missing_materials},
            )
    else:
        vertices = 0
        faces = 0
        materials = 1
    result = {
        "file": file_row["path"],
        "format": path.suffix.lower().lstrip("."),
        "size": path.stat().st_size,
        "vertices": vertices,
        "faces": faces,
        "materials": materials,
    }
    ctx.log(
        f"Validated model {path.name} ({result['size']} bytes, {vertices} vertices, {faces} faces)",
        stage="MESH_VALIDATION",
        operation="validate",
        status="completed",
    )
    return result
