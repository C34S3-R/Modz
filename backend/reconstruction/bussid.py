"""Phase 9: BUSSID preparation and the ``.bussidmod`` package.

Mirrors the web backend's two stages without needing the database:

- ``pipeline/bussid.py`` - copy the prepared model plus its referenced
  sidecars into ``bussid/`` and write a ``manifest.json`` there.
- ``pipeline/exporter.py`` - ZIP the deliverables into
  ``exports/<slug>.bussidmod`` (root ``manifest.json`` with
  ``format: bussidmod``, files under ``assets/...``), atomically via a
  ``.part`` file, then write ``exports/result.json``.

Two deliberate differences from the web exporter:

- **Only the deliverables are packaged** (model + MTL + texture +
  manifests).  The web exporter sweeps whole workspace roots because it
  must package whatever previous jobs registered; the engine knows
  exactly what BUSSID consumes, so intermediates (dense maps, raw meshes)
  can never leak into a mod that a phone has to import.
- **Fixed entry timestamps.**  ``zipfile`` otherwise records each
  source file's mtime, so re-packaging identical bytes would produce a
  different archive and fail the determinism check.  Entries are written
  with a constant 1980-01-01 timestamp instead.
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ReconConfig, ReconstructionError
from .state import StageContext, StateStore, memory_pressure

# A BUSSID mod is a few MB (obj + 1024^2 PNG); a package this large means
# intermediates leaked in or a texture went runaway, so fail honestly
# instead of filling the HDD.  (The web exporter caps at 8 GiB because it
# packages whole workspace roots.)
_MAX_PACKAGE_BYTES = 512 * 1024 * 1024

# Sidecar kinds the OBJ/MTL chain can reference.
_MTL_SUFFIXES = {".mtl"}
_TEXTURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tga", ".bmp"}


def _referenced_mtls(obj: Path) -> List[str]:
    """MTL names from the OBJ's ``mtllib`` lines, in order, deduplicated."""
    names: List[str] = []
    for line in obj.read_text(encoding="utf-8",
                              errors="replace").splitlines():
        if line.startswith("mtllib "):
            name = line.split(None, 1)[1].strip()
            if name and name not in names:
                names.append(name)
    return names


def _texture_map_lines(mtl_text: str) -> List[Tuple[int, str, str]]:
    """(line index, keyword, filename) for every texture-map directive.

    Handles option-carrying forms like ``map_Kd -s 1 1 1 tex.png`` by
    taking the last token as the filename.
    """
    found: List[Tuple[int, str, str]] = []
    for index, line in enumerate(mtl_text.splitlines()):
        stripped = line.strip()
        lowered = stripped.lower()
        keyword = lowered.split(None, 1)[0] if stripped else ""
        if keyword.startswith("map_") or keyword in {"bump", "disp", "norm"}:
            tokens = stripped.split()
            if len(tokens) >= 2:
                found.append((index, tokens[0], tokens[-1]))
    return found


def _resolve_sidecar(name: str, *dirs: Path) -> Path:
    """Find a referenced sidecar: as given, then by basename in each dir."""
    candidate = Path(name)
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate
    else:
        for base in dirs:
            direct = base / candidate
            if direct.is_file():
                return direct
    for base in dirs:
        flat = base / candidate.name
        if flat.is_file():
            return flat
    raise ReconstructionError(
        f"BUSSID preparation found a referenced file missing: {name}",
        suggestion="The model's material references a texture that does "
                   "not exist next to it; re-run the texture and blender "
                   "stages so the sidecar set is rebuilt.",
        details={"referenced": name,
                 "searched": [str(d) for d in dirs]},
    )


def _prepare_dir(cfg: ReconConfig, model: Path, out_dir: Path
                 ) -> Tuple[List[str], Path]:
    """Copy model + referenced sidecars into ``bussid/``, write manifest.

    Returns (asset names, manifest path).  The directory is rebuilt from
    scratch so stale files from an earlier layout cannot be swept into
    the package.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    mtl_names = _referenced_mtls(model)
    if not mtl_names:
        raise ReconstructionError(
            f"{model.name} references no material file (no mtllib line).",
            suggestion="Re-run the texture stage (and blender stage); both "
                       "guarantee the OBJ -> MTL link.",
            details={"model": str(model)},
        )

    texture_dir = cfg.d("texture")
    blender_dir = cfg.d("blender")
    names: List[str] = []
    seen: set = set()          # guards against duplicate zip entries
    mtl_texts: Dict[str, str] = {}

    destination = out_dir / model.name
    shutil.copy2(model, destination)
    names.append(model.name)
    seen.add(model.name)

    # Flat assembly: every sidecar lands next to the model under its
    # basename, and the MTL copy is rewritten to match.
    pending: List[Tuple[str, Path]] = []
    for mtl_name in mtl_names:
        source = _resolve_sidecar(mtl_name, model.parent, texture_dir,
                                  blender_dir)
        text = source.read_text(encoding="utf-8", errors="replace")
        flat_names: Dict[int, str] = {}
        for index, _keyword, filename in _texture_map_lines(text):
            texture_source = _resolve_sidecar(
                filename, source.parent, texture_dir, blender_dir)
            flat_names[index] = texture_source.name
            if texture_source.name not in names:
                pending.append((texture_source.name, texture_source))
        if flat_names:
            lines = text.splitlines()
            for index, flat in flat_names.items():
                tokens = lines[index].split()
                lines[index] = " ".join(tokens[:-1] + [flat])
            text = "\n".join(lines) + "\n"
        if source.name not in mtl_texts:
            mtl_texts[source.name] = text

    for mtl_name, text in mtl_texts.items():
        if mtl_name in seen:
            continue
        (out_dir / mtl_name).write_text(text, encoding="utf-8")
        names.append(mtl_name)
        seen.add(mtl_name)
    for flat, source in pending:
        if flat in seen:
            continue
        shutil.copy2(source, out_dir / flat)
        names.append(flat)
        seen.add(flat)

    manifest = {
        "format": "bussid",
        "model": model.name,
        "assets": names,
        "lights": [],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2),
                             encoding="utf-8")
    return names, manifest_path


def _write_entry(archive: zipfile.ZipFile, path: Path, arcname: str) -> int:
    """Add a file with a fixed timestamp so identical bytes -> identical zip."""
    info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3              # unix
    info.external_attr = 0o644 << 16
    with path.open("rb") as handle:
        data = handle.read()
    archive.writestr(info, data)
    return len(data)


def _package(cfg: ReconConfig, out_dir: Path, names: List[str],
             ctx: StageContext) -> Dict[str, Any]:
    """Build ``exports/<slug>.bussidmod`` atomically (exporter.py pattern)."""
    exports = cfg.d("exports")
    exports.mkdir(parents=True, exist_ok=True)
    slug = cfg.project_dir.name
    package_path = exports / f"{slug}.bussidmod"
    temporary = package_path.with_suffix(package_path.suffix + ".part")
    temporary.unlink(missing_ok=True)

    assets = [f"assets/bussid/{name}" for name in names]
    manifest = {
        "format": "bussidmod",
        "project": slug,
        "model": names[0],
        "assets": assets,
    }
    try:
        with zipfile.ZipFile(temporary, "w",
                             compression=zipfile.ZIP_DEFLATED) as archive:
            manifest_info = zipfile.ZipInfo("manifest.json",
                                            date_time=(1980, 1, 1, 0, 0, 0))
            manifest_info.compress_type = zipfile.ZIP_DEFLATED
            manifest_info.create_system = 3
            manifest_info.external_attr = 0o644 << 16
            archive.writestr(manifest_info, json.dumps(manifest, indent=2))
            for name in names:
                ctx.check()
                _write_entry(archive, out_dir / name,
                             f"assets/bussid/{name}")
                if temporary.stat().st_size > _MAX_PACKAGE_BYTES:
                    raise ReconstructionError(
                        "BUSSID package exceeds the size limit.",
                        suggestion="Intermediates leaked into bussid/ or the "
                                   "texture is far larger than the preset; "
                                   "inspect bussid/ and re-run the stage.",
                        details={"limit_bytes": _MAX_PACKAGE_BYTES},
                        exit_code=1,
                    )
        temporary.replace(package_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    result = {
        "filename": package_path.name,
        "path": str(package_path.relative_to(cfg.project_dir)),
        "download_url": None,          # the web backend builds its own URL
        "size": package_path.stat().st_size,
        "asset_count": len(names),
        "project": slug,
    }
    (exports / "result.json").write_text(json.dumps(result, indent=2),
                                         encoding="utf-8")
    return result


def run_bussid(cfg: ReconConfig, store: StateStore,
               ctx: StageContext) -> Dict[str, Any]:
    started = time.time()
    ctx.check()

    # The prepared model comes from the blender stage (either the real
    # cleanup or its model-copy fallback - both write blender_meta.json).
    model: Optional[Path] = None
    meta_path = cfg.d("blender", "blender_meta.json")
    if meta_path.is_file():
        try:
            recorded = json.loads(
                meta_path.read_text(encoding="utf-8")).get("output")
            if recorded:
                candidate = cfg.project_dir / recorded
                if candidate.is_file():
                    model = candidate
        except (OSError, ValueError, TypeError):
            model = None
    if model is None:
        candidates = sorted(cfg.d("blender").glob("prepared_*.obj")) \
            if cfg.d("blender").is_dir() else []
        model = candidates[0] if len(candidates) == 1 else None
    if model is None or not model.is_file():
        raise ReconstructionError(
            "BUSSID stage needs the prepared model from the blender stage "
            "(blender/prepared_*.obj).",
            suggestion="Run the pipeline through the 'blender' stage first.",
            details={"project": str(cfg.project_dir)},
        )

    ctx.set_command(
        f"prep bussid/ from {model.name} and package {cfg.project_dir.name}"
        f".bussidmod")
    out_dir = cfg.d("bussid")
    names, manifest_path = _prepare_dir(cfg, model, out_dir)
    ctx.check()

    package = _package(cfg, out_dir, names, ctx)
    ctx.set_output(exit_code=0)

    meta: Dict[str, Any] = {
        "fingerprint": cfg.bussid_fingerprint(),
        "source": str(model.relative_to(cfg.project_dir)),
        "assets": names,
        "manifest": str(manifest_path.relative_to(cfg.project_dir)),
        "package": package,
        "elapsed_seconds": round(time.time() - started, 2),
        "memory_peak_gb": memory_pressure(cfg.ram_limit_gb)["process_peak_gb"],
    }
    bussid_meta = out_dir / "bussid_meta.json"
    bussid_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    outputs = [out_dir / name for name in names] + [
        bussid_meta,
        cfg.d("exports") / package["filename"],
        cfg.d("exports") / "result.json",
    ]
    ctx.add_output_files(outputs)
    ctx.log(
        f"packaged {package['filename']} ({package['asset_count']} assets, "
        f"{package['size']} bytes) in {meta['elapsed_seconds']}s",
        operation="complete",
    )
    result = {
        "model": f"bussid/{model.name}",
        "manifest": str(manifest_path.relative_to(cfg.project_dir)),
        "package": package["path"],
        "package_size": package["size"],
        "asset_count": package["asset_count"],
        "elapsed_seconds": meta["elapsed_seconds"],
    }
    ctx.result(result)
    return result
