"""Model export and BUSSID package creation."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .common import PipelineContext, StageError


def export_model(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.check()
    model = ctx.model_file()
    if not model:
        raise StageError("EXPORT FAILED: no model is available to export")
    file_row, source = model
    export_dir = ctx.project_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    destination = export_dir / f"{ctx.project['slug']}_preview{source.suffix.lower()}"
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    metadata = {
        "filename": destination.name,
        "model": source.name,
        "format": source.suffix.lower().lstrip("."),
        "size": destination.stat().st_size,
    }
    metadata_path = export_dir / "export_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ctx.register(destination, "model")
    ctx.register(metadata_path, "metadata")
    return metadata


def _current_assets(ctx: PipelineContext) -> Tuple[list[Tuple[Path, str]], Optional[Path]]:
    """Return only current-revision model assets, never the whole workspace."""

    model = ctx.model_file()
    if not model:
        return [], None
    _model_row, selected_model = model
    current_revision = int(ctx.project.get("revision") or 1)
    current_paths = {
        row["path"]
        for row in ctx.file_manager.raw_rows_for_project(ctx.project_id)
        if int(row.get("revision") or 1) == current_revision
    }
    assets: list[Tuple[Path, str]] = []
    for root_name in ("bussid", "blender", "reconstruction", "lights"):
        root = ctx.project_dir / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(ctx.project_dir).as_posix()
            # Generated files are registered by pipeline stages.  The fallback
            # keeps useful sidecars from older/manual projects packageable.
            if relative not in current_paths and current_paths:
                continue
            assets.append((path, f"assets/{relative}"))
    if selected_model not in [path for path, _ in assets]:
        assets.append((selected_model, f"assets/{selected_model.name}"))
    return assets, selected_model


def package_result(ctx: PipelineContext) -> Dict[str, Any]:
    # Build into a temporary archive first.  A cancelled or failed export must
    # not leave a file that looks like a valid final package.
    ctx.check()
    assets, selected_model = _current_assets(ctx)
    if not assets or selected_model is None:
        raise StageError("PACKAGE FAILED: no model assets are available to package")
    export_dir = ctx.project_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    # General 3D projects package as .zip; only projects with the BUSSID
    # game-export option enabled produce a .bussidmod (same zip layout).
    bussid_enabled = bool((ctx.project.get("options") or {}).get("bussid", False))
    extension = ".bussidmod" if bussid_enabled else ".zip"
    package_path = export_dir / f"{ctx.project['slug']}{extension}"
    # Rebuild the archive atomically so a cancelled/failed export never leaves
    # a misleading final-looking package at the target path.
    temporary = package_path.with_suffix(package_path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    manifest = {
        "format": "bussidmod" if bussid_enabled else "modzip",
        "project": ctx.project.get("name"),
        "project_type": ctx.project.get("project_type") or "general",
        "model": selected_model.name,
        "assets": [archive_name for _path, archive_name in assets],
    }
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
            for path, archive_name in assets:
                ctx.check()
                if path.is_file() and not path.is_symlink():
                    archive.write(path, archive_name)
                    if temporary.stat().st_size > ctx.config.max_export_bytes:
                        raise StageError(
                            "PACKAGE FAILED: export exceeds the configured size limit",
                            exit_code=1,
                        )
        if temporary.stat().st_size > ctx.config.max_export_bytes:
            raise StageError(
                "PACKAGE FAILED: export exceeds the configured size limit",
                exit_code=1,
            )
        temporary.replace(package_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    registered = ctx.register(package_path, "export")
    result = {
        "filename": package_path.name,
        "path": registered["path"],
        "download_url": f"/api/projects/{ctx.project_id}/files/download?path={registered['path']}",
        "size": package_path.stat().st_size,
        "asset_count": len(assets),
    }
    ctx.write_json("exports", "result.json", result)
    ctx.register(ctx.project_dir / "exports" / "result.json", "metadata")
    # Keep a copy in the configured global output directory for integrations
    # that consume server-level outputs rather than project-relative paths.
    try:
        ctx.config.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_path, ctx.config.output_dir / package_path.name)
    except OSError as exc:
        warning = f"Global output mirror failed: {exc}"
        result["warnings"] = [warning]
        ctx.write_json("exports", "result.json", result)
        ctx.log(warning, stage="PACKAGE", operation="mirror", status="warning", level="WARNING")
    ctx.log(
        f"Packaged {package_path.name} ({result['size']} bytes, {len(assets)} assets)",
        stage="PACKAGE",
        operation="package",
        status="completed",
    )
    return result
