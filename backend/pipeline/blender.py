"""Blender integration with a safe built-in fallback for model-only jobs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .common import PipelineContext, StageError, referenced_sidecars
from utils.subprocess import CommandError, run_command

_MODEL_EXTS = {".blend", ".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply"}


def _prepare_source(ctx: PipelineContext
                    ) -> Optional[Tuple[Dict[str, Any], Path]]:
    """Input model for this stage: never the stage's own previous output.

    ``model_file()`` prefers ``blender/`` so *downstream* stages consume the
    prepared copy, but on a retry that would feed blender/ straight back into
    itself: the stale copy would be re-prepared, and its sidecar would resolve
    to the staging destination (same-file copy crash).  Prefer the fresh
    reconstruction, then the uploaded input, then any registered model.
    """
    revision = int(ctx.project.get("revision") or 1)
    for roots in (["reconstruction"], ["input"]):
        found = ctx.files(roots=roots, extensions=_MODEL_EXTS,
                          revision=revision)
        if found:
            return found[0]
    return ctx.model_file()


def prepare(ctx: PipelineContext) -> Dict[str, Any]:
    # FastAPI launches Blender as a child process; it never imports Blender or
    # performs mesh operations in the web request thread.
    ctx.check()
    model = _prepare_source(ctx)
    if not model:
        raise StageError("BLENDER FAILED: no model is available to prepare")
    file_row, source = model
    blender = ctx.settings.tool("blender_path")
    output_dir = ctx.project_dir / "blender"
    output_dir.mkdir(parents=True, exist_ok=True)
    if blender:
        script = Path(__file__).resolve().parent.parent / "blender_scripts" / "prepare_model.py"
        output = output_dir / f"prepared_{source.stem}.blend"
        command = [
            blender,
            "--background",
            "--python",
            str(script),
            "--",
            "--input",
            str(source),
            "--output",
            str(output),
        ]
        try:
            run_command(
                command,
                cwd=ctx.project_dir,
                log_path=ctx.tool_log("blender"),
                control=ctx.control,
                timeout_seconds=ctx.config.command_timeout_seconds,
            )
        except CommandError as exc:
            raise StageError(
                "BLENDER FAILED: model preparation process returned an error",
                exit_code=exc.returncode,
                details={"command": exc.command, "output": exc.output[-6000:]},
            ) from exc
        if not output.is_file() or output.stat().st_size == 0:
            raise StageError("BLENDER FAILED: Blender did not produce a prepared model", exit_code=1)
        preview = output.with_suffix(".glb")
        selected_output = preview if preview.is_file() and preview.stat().st_size else output
        if selected_output is preview:
            ctx.register(selected_output, "model")
        result = {"model": f"blender/{selected_output.name}", "fallback": False}
    else:
        if ctx.config.require_external_tools:
            raise StageError(
                "BLENDER FAILED: Blender is not installed or configured",
                exit_code=127,
            )
        output = output_dir / f"prepared_{source.name}"
        shutil.copy2(source, output)
        # Carry the model's referenced material/texture chain verbatim: the
        # copy keeps its mtllib line, so a lone OBJ would be textureless in
        # every consumer that resolves references beside the model.
        for sidecar in referenced_sidecars(source):
            sidecar_output = output_dir / sidecar.name
            if sidecar.resolve() != sidecar_output.resolve():
                shutil.copy2(sidecar, sidecar_output)
            ctx.register(
                sidecar_output,
                "metadata" if sidecar.suffix.lower() in {".mtl", ".bin"} else "texture",
            )
        result = {"model": "blender/" + output.name, "fallback": True}
        ctx.log(
            "Blender is unavailable; preserved the input model as a recoverable compatibility copy",
            stage="BLENDER",
            operation="fallback",
            status="warning",
            level="WARNING",
        )
    ctx.register(output, "model")
    ctx.log(
        f"Prepared model {source.name}",
        stage="BLENDER",
        operation="prepare",
        status="completed",
    )
    return result
