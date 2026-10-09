"""Blender-side finalization for the reconstruction engine.

Run through Blender with
``--background --python finalize_model.py -- --input ... --output ...``.

Imports the textured OBJ, recalculates outward normals and re-exports an
OBJ + MTL pair.  Two deliberate choices:

- **No welding.**  The texture stage stores per-corner UVs and chart
  seams share vertex positions, so merge-by-distance would destroy the
  atlas layout that was just built.  Cleanup here means normals only;
  geometry quality gates already ran upstream (mesh/lowpoly stages).
- **Version tolerance.**  Import/export operator names moved between
  Blender releases (``import_scene.obj`` -> ``wm.obj_export``), so each
  call probes for the new name first, like ``prepare_model.py``.

The engine normalizes the MTL/texture references afterwards, so this
script only has to produce valid geometry; exporter quirks around
``map_Kd`` paths cannot fail the stage.
"""

import argparse
import sys
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(
        sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def import_obj(source: Path) -> None:
    import bpy

    importer = getattr(bpy.ops.wm, "obj_import", None)
    if importer is not None:
        importer(filepath=str(source))
    else:
        bpy.ops.import_scene.obj(filepath=str(source))


def recalc_normals() -> None:
    """Outward normals for every mesh object; best-effort across builds."""
    import bpy

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        return
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        # Some builds rename or restrict these operators.  Missing vn
        # entries are cosmetic (importers recompute), so never fail here.
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass


def export_obj(destination: Path) -> None:
    import bpy

    exporter = getattr(bpy.ops.wm, "obj_export", None)      # Blender 4.x
    if exporter is not None:
        try:
            exporter(filepath=str(destination), export_materials=True,
                     export_uv=True, export_normals=True)
            return
        except TypeError:
            pass  # keyword drift between 4.x minors: retry with defaults
    legacy = getattr(getattr(bpy.ops, "export_scene", None), "obj", None)
    if legacy is not None:
        legacy(filepath=str(destination), use_selection=False,
               write_material=True, path_mode="COPY")
        return
    raise RuntimeError("This Blender build has no OBJ exporter")


def main():
    args = arguments()
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError("This script must run inside Blender") from exc

    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".obj":
        raise RuntimeError(
            f"finalize_model.py expects an OBJ input, got {source.suffix!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_obj(source)
    if not bpy.context.scene.objects:
        raise RuntimeError("Imported model contains no objects")
    recalc_normals()
    export_obj(destination)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("OBJ export produced no output")


if __name__ == "__main__":
    main()
