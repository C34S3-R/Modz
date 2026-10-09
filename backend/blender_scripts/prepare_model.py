"""Blender-side model preparation script.

Run through Blender with ``--background --python prepare_model.py -- --input ...``.
The script deliberately keeps the original input untouched and writes a new
prepared model.  It is intentionally conservative: unsupported formats are
reported to Blender's process exit status instead of being silently discarded.
"""

import argparse
import sys
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if suffix == ".obj":
        importer = getattr(bpy.ops.wm, "obj_import", None)
        if importer is None:
            bpy.ops.import_scene.obj(filepath=str(source))
        else:
            importer(filepath=str(source))
    elif suffix == ".stl":
        importer = getattr(bpy.ops.wm, "stl_import", None)
        if importer is None:
            bpy.ops.import_scene.stl(filepath=str(source))
        else:
            importer(filepath=str(source))
    elif suffix == ".ply":
        importer = getattr(bpy.ops.wm, "ply_import", None)
        if importer is None:
            bpy.ops.import_mesh.ply(filepath=str(source))
        else:
            importer(filepath=str(source))
    elif suffix in {".fbx", ".glb", ".gltf"}:
        bpy.ops.import_scene.fbx(filepath=str(source)) if suffix == ".fbx" else bpy.ops.import_scene.gltf(filepath=str(source))
    elif suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(source))
    else:
        raise RuntimeError(f"Unsupported model format: {suffix}")
    if not bpy.context.scene.objects:
        raise RuntimeError("Imported model contains no objects")
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    if destination.suffix.lower() == ".blend":
        try:
            bpy.ops.export_scene.gltf(
                filepath=str(destination.with_suffix(".glb")),
                export_format="GLB",
            )
        except Exception:
            # Older Blender builds may not include the glTF exporter; the
            # .blend checkpoint remains valid and can be exported later.
            pass


if __name__ == "__main__":
    main()
