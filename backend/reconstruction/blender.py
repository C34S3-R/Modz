"""Phase 8: optional Blender cleanup with an honest model-copy fallback.

Prepares the textured model for packaging.  When Blender is installed
(``BLENDER_PATH`` or ``blender`` on ``PATH``) the stage runs
``blender_scripts/finalize_model.py`` headless: import the OBJ,
recalculate outward normals, re-export.  When Blender is missing the
stage copies the textured model through unchanged and says so loudly -
the same contract as the web pipeline's Blender stage (AGENTS: "logged
model-copy fallback unless ``REQUIRE_EXTERNAL_TOOLS=true`"`).

Either way the output directory is self-contained: this module pins the
``OBJ -> MTL -> texture.png`` chain itself, because exporters differ in
how they name the MTL and write ``map_Kd`` (absolute paths, ``//``
prefixes, sometimes nothing at all).  A broken reference would only
surface when BUSSID loads the mod, which is exactly where a silent
failure hurts most.

No lights here: light objects belong to the web pipeline's
analysis/creation stages (database-backed), not to the standalone
engine.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from .config import ReconConfig, ReconstructionError, resolve_blender
from .state import StageContext, StateStore, memory_pressure
from utils.subprocess import CommandError, run_command

_TEXTURE_SIDECARS = (".mtl", ".png", ".jpg", ".jpeg", ".webp", ".tga", ".bmp")


def _finalize_sidecars(cfg: ReconConfig, out: Path, out_dir: Path
                       ) -> List[Path]:
    """Guarantee the ``OBJ -> MTL -> texture.png`` chain inside ``out_dir``.

    Returns the sidecar paths that were ensured (already-present files
    with the right content are left untouched, so their mtimes - and
    therefore the packaged ZIP - stay stable across re-runs).
    """
    texture_dir = cfg.d("texture")
    lines = out.read_text(encoding="utf-8", errors="replace").splitlines()
    mtllib = next((line.split(None, 1)[1].strip()
                   for line in lines if line.startswith("mtllib ")), None)
    if mtllib is None or Path(mtllib).name != mtllib:
        # Missing entirely (a Blender build exported without materials),
        # or referencing a subdirectory - the files are assembled flat,
        # so pin the OBJ to the flat name this function creates.
        flat = Path(mtllib).name if mtllib else "textured.mtl"
        rewritten = []
        replaced = False
        for line in lines:
            if line.startswith("mtllib "):
                rewritten.append("mtllib " + flat)
                replaced = True
            else:
                rewritten.append(line)
        if not replaced:
            rewritten.insert(0, "mtllib " + flat)
        out.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
        mtllib = flat

    mtl_path = out_dir / Path(mtllib).name      # never allow path escape
    if not mtl_path.is_file():
        source_mtl = texture_dir / "textured.mtl"
        if not source_mtl.is_file():
            raise ReconstructionError(
                "The textured model's material file is missing "
                "(texture/textured.mtl).",
                suggestion="Re-run the texture stage; it writes the MTL "
                           "alongside texture.png.",
                details={"expected": str(source_mtl)},
            )
        shutil.copy2(source_mtl, mtl_path)

    # Pin the texture reference: one map_Kd line, pointing at the PNG
    # that sits next to it.  Rewrite only when the content differs, so
    # a re-run never bumps the file's mtime.
    mtl_lines = mtl_path.read_text(encoding="utf-8",
                                   errors="replace").splitlines()
    changed = False
    found_map = False
    for index, line in enumerate(mtl_lines):
        if line.strip().startswith("map_Kd"):
            if line.strip() != "map_Kd texture.png":
                mtl_lines[index] = "map_Kd texture.png"
                changed = True
            found_map = True
    if not found_map:
        for index, line in enumerate(mtl_lines):
            if line.strip().startswith("newmtl"):
                mtl_lines.insert(index + 1, "map_Kd texture.png")
                changed = True
                break
        else:
            mtl_lines += ["newmtl vehicle", "map_Kd texture.png"]
            changed = True
    if changed:
        mtl_path.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")

    png_path = out_dir / "texture.png"
    if not png_path.is_file():
        source_png = texture_dir / "texture.png"
        if not source_png.is_file():
            raise ReconstructionError(
                "The baked texture image is missing (texture/texture.png).",
                suggestion="Re-run the texture stage; it writes the PNG "
                           "the material references.",
                details={"expected": str(source_png)},
            )
        shutil.copy2(source_png, png_path)
    return [mtl_path, png_path]


def run_blender(cfg: ReconConfig, store: StateStore,
                ctx: StageContext) -> Dict[str, Any]:
    src = cfg.d("texture", "textured.obj")
    if not src.is_file():
        raise ReconstructionError(
            "Blender stage needs the textured model (texture/textured.obj).",
            suggestion="Run the pipeline through the 'texture' stage first.",
            details={"project": str(cfg.project_dir)},
        )
    out_dir = cfg.d("blender")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"prepared_{src.name}"
    started = time.time()
    ctx.check()

    tool = resolve_blender(cfg.blender_path)
    if tool is None:
        if cfg.require_external_tools:
            raise ReconstructionError(
                "Blender is not installed or configured and "
                "REQUIRE_EXTERNAL_TOOLS=true forbids the fallback.",
                suggestion="Install Blender and set BLENDER_PATH, or unset "
                           "the strict policy to allow the model-copy "
                           "fallback.",
                details={"blender_path": cfg.blender_path},
                exit_code=127,
            )
        command = (f"model-copy fallback (Blender unavailable): "
                   f"{src.name} -> {out.name}")
        ctx.set_command(command)
        shutil.copy2(src, out)
        fallback = True
        ctx.log(
            "Blender is unavailable; preserved the textured model as a "
            "recoverable compatibility copy",
            operation="fallback", status="warning",
        )
    else:
        script = (Path(__file__).resolve().parents[1]
                  / "blender_scripts" / "finalize_model.py")
        command_list = [
            tool, "--background", "--factory-startup", "--python",
            str(script), "--",
            "--input", str(src), "--output", str(out),
        ]
        command = " ".join(command_list)
        ctx.set_command(command_list)
        log_path = cfg.d("logs", "blender.log")
        try:
            run_command(command_list, cwd=cfg.project_dir,
                        log_path=log_path, control=ctx.control,
                        timeout_seconds=1800)
        except CommandError as exc:
            raise ReconstructionError(
                "Blender finalization failed while preparing the textured "
                "model.",
                suggestion="Check logs/blender.log; set BLENDER_PATH to a "
                           "working Blender, or unset REQUIRE_EXTERNAL_TOOLS "
                           "to allow the model-copy fallback.",
                details={"exit_code": exc.returncode,
                         "command": command_list,
                         "output": (exc.output or "")[-800:]},
                exit_code=exc.returncode,
            ) from exc
        if not out.is_file() or out.stat().st_size == 0:
            raise ReconstructionError(
                "Blender did not produce the prepared model.",
                suggestion="Check logs/blender.log for exporter errors.",
                details={"expected": str(out)},
                exit_code=1,
            )
        fallback = False

    ctx.check()
    sidecars = _finalize_sidecars(cfg, out, out_dir)
    ctx.set_output(exit_code=0)

    meta: Dict[str, Any] = {
        "fingerprint": cfg.blender_fingerprint(),
        "tool": tool,
        "fallback": fallback,
        "command": command,
        "exit_code": 0,
        "source": str(src.relative_to(cfg.project_dir)),
        "output": str(out.relative_to(cfg.project_dir)),
        "sidecars": [path.name for path in sidecars],
        "cleaned": ["normals"] if not fallback else [],
        "elapsed_seconds": round(time.time() - started, 2),
        "memory_peak_gb": memory_pressure(cfg.ram_limit_gb)["process_peak_gb"],
    }
    meta_path = out_dir / "blender_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    outputs = [out, meta_path, *sidecars]
    ctx.add_output_files(outputs)
    ctx.log(
        f"{'finalized with ' + Path(tool).name if not fallback else 'fallback copy of'}"
        f" {src.name} -> {out.name} "
        f"({meta['elapsed_seconds']}s)",
        operation="complete", status="info" if not fallback else "warning",
    )
    result = {
        "model": f"blender/{out.name}",
        "fallback": fallback,
        "tool": tool,
        "sidecars": [path.name for path in sidecars],
        "elapsed_seconds": meta["elapsed_seconds"],
    }
    ctx.result(result)
    return result
