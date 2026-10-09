"""Safe filesystem helpers for untrusted uploads and project-relative paths."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional


class FileValidationError(ValueError):
    pass


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_EXTENSIONS = {".obj", ".fbx", ".glb", ".gltf", ".blend", ".stl", ".ply"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS | MODEL_EXTENSIONS

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: Optional[str], fallback: str = "upload.bin") -> str:
    """Return a basename that cannot contain path separators or traversal."""

    raw = str(filename or "").replace("\\", "/")
    raw = PurePosixPath(raw).name
    raw = unicodedata.normalize("NFKC", raw)
    raw = _SAFE_CHARS.sub("_", raw).strip("._ ")
    if not raw:
        raw = fallback
    # Keep the extension while limiting the total path component length.
    if len(raw) > 180:
        suffix = Path(raw).suffix[:16]
        stem = raw[: max(1, 180 - len(suffix))]
        raw = stem + suffix
    return raw


def extension_for(filename: str) -> str:
    return Path(filename).suffix.lower()


def file_kind(filename: str) -> str:
    extension = extension_for(filename)
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in MODEL_EXTENSIONS:
        return "model"
    return "unknown"


def validate_upload_name(filename: Optional[str]) -> tuple[str, str]:
    safe = sanitize_filename(filename)
    extension = extension_for(safe)
    kind = file_kind(safe)
    if kind == "unknown" or extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise FileValidationError(
            f"Unsupported file type '{extension or '(none)'}'. Allowed extensions: {allowed}"
        )
    return safe, kind


def validate_mime_type(content_type: Optional[str], kind: str) -> None:
    """Allow browser MIME variance while rejecting clearly unrelated types."""

    if not content_type:
        return
    normalized = content_type.split(";", 1)[0].strip().lower()
    if normalized in {"application/octet-stream", "binary/octet-stream"}:
        return
    prefixes = {
        "video": ("video/",),
        "image": ("image/",),
        "model": ("model/", "application/obj", "application/octet-stream"),
    }
    if normalized.startswith(prefixes.get(kind, ())):
        return
    # Some browsers report these generic types for model files.
    if kind == "model" and normalized in {
        "application/x-obj",
        "application/octet-stream",
        "application/zip",
    }:
        return
    raise FileValidationError(
        f"MIME type '{content_type}' does not match a {kind} upload"
    )


def unique_destination(directory: Path, filename: str) -> Path:
    """Choose a non-colliding path while retaining the original extension."""

    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def resolve_within(root: Path, relative_path: str | PurePosixPath) -> Path:
    """Resolve a project-relative path and reject absolute paths/traversal."""

    root = root.resolve()
    raw = str(relative_path).replace("\\", "/")
    if not raw or raw.startswith("/") or ":" in raw.split("/", 1)[0]:
        raise FileValidationError("A project-relative path is required")
    candidate = (root / PurePosixPath(raw)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FileValidationError("Path escapes the project directory") from exc
    return candidate


def relative_to_root(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def iter_files(root: Path, extensions: Optional[Iterable[str]] = None) -> Iterable[Path]:
    allowed = {e.lower() for e in extensions} if extensions else None
    if not root.exists():
        return []
    return (
        path
        for path in root.rglob("*")
        if path.is_file() and (allowed is None or path.suffix.lower() in allowed)
    )
