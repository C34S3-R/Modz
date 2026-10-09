"""Server-side import of reference videos/images from online URLs.

Stdlib only (urllib) so no new dependencies.  Every URL is validated
before a single byte is fetched:

* scheme must be http or https, no embedded credentials;
* the host must resolve to a public IP - loopback, private, link-local,
  multicast, reserved and unspecified addresses are refused (SSRF guard);
* the final URL after redirects is re-checked the same way;
* the payload must look like a video/image/model (extension allowlist,
  same as uploads) and stay under the project's upload byte limit.

YouTube watch/shorts/share links are resolved with yt-dlp (direct media
URLs expire and are IP-bound, so the page URL itself is the stable
input); everything else is fetched as a plain file download.
"""

from __future__ import annotations

import mimetypes
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.filesystem import (
    ALLOWED_EXTENSIONS,
    FileValidationError,
    file_kind,
    sanitize_filename,
    unique_destination,
    validate_upload_name,
)


class URLImportError(ValueError):
    pass


YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}


def is_youtube_url(parsed: urllib.parse.ParseResult) -> bool:
    return (parsed.hostname or "").lower().rstrip(".") in YOUTUBE_HOSTS


_USER_AGENT = "ModzVehicleServer/1.0 (reference-media import)"
_CONNECT_TIMEOUT = 15
_READ_TIMEOUT = 60

# Content types we accept per file kind.  Servers often serve models and
# some videos as application/octet-stream, so that stays allowed.
_VIDEO_TYPES = ("video/",)
_IMAGE_TYPES = ("image/",)
_GENERIC_TYPES = {
    "application/octet-stream",
    "binary/octet-stream",
}


def validate_source_url(raw: str) -> urllib.parse.ParseResult:
    """Parse and SSRF-guard a candidate source URL."""
    text = (raw or "").strip()
    if not text:
        raise URLImportError("An online source URL is required")
    if len(text) > 2048:
        raise URLImportError("The source URL is too long")
    try:
        parsed = urllib.parse.urlparse(text)
    except ValueError as exc:
        raise URLImportError(f"Malformed source URL: {exc}") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise URLImportError("Only http:// and https:// source URLs are supported")
    if parsed.username or parsed.password:
        raise URLImportError("Source URLs must not embed credentials")
    host = parsed.hostname or ""
    if not host:
        raise URLImportError("The source URL has no host")
    _reject_non_public_host(host)
    return parsed


def _reject_non_public_host(host: str) -> None:
    lowered = host.lower().rstrip(".")
    if lowered == "localhost":
        raise URLImportError("Local hosts are not valid online sources")
    try:
        addresses = socket.getaddrinfo(host, None, family=socket.AF_UNSPEC)
    except socket.gaierror as exc:
        raise URLImportError(f"Could not resolve the source host '{host}'") from exc
    seen = set()
    for family, _socktype, _proto, _canon, sockaddr in addresses:
        ip_text = sockaddr[0]
        if ip_text in seen:
            continue
        seen.add(ip_text)
        try:
            ip = ip_address(ip_text)
        except ValueError:
            raise URLImportError(f"Unresolvable source host '{host}'")
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise URLImportError(
                f"Source host '{host}' resolves to a non-public address"
            )


def _filename_for_url(
    parsed: urllib.parse.ParseResult,
    content_type: Optional[str],
    content_disposition: Optional[str],
) -> str:
    candidate = ""
    if content_disposition:
        for part in content_disposition.split(";"):
            part = part.strip()
            if part.lower().startswith("filename*="):
                candidate = part.split("=", 1)[1].strip().strip("\"'")
                if "''" in candidate:
                    candidate = urllib.parse.unquote(candidate.split("''", 1)[1])
                break
            if part.lower().startswith("filename="):
                candidate = part.split("=", 1)[1].strip().strip("\"'")
                break
    if not candidate:
        candidate = Path(urllib.parse.unquote(parsed.path or "")).name
    safe = sanitize_filename(candidate, fallback="")
    if Path(safe).suffix.lower() in ALLOWED_EXTENSIONS:
        validate_upload_name(safe)
        return safe
    # URL path carried no usable media name: derive one from content type.
    guessed = (mimetypes.guess_extension((content_type or "").split(";")[0].strip().lower()) or "").lower()
    if guessed not in ALLOWED_EXTENSIONS:
        # Map the common cases guess_extension misses.
        fallback = {
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/x-matroska": ".mkv",
            "image/jpeg": ".jpg",
        }.get((content_type or "").split(";")[0].strip().lower(), "")
        guessed = fallback
    if guessed not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise URLImportError(
            f"Online source is not a supported media type (got '{content_type or 'unknown'}'). "
            f"Allowed extensions: {allowed}"
        )
    stem = Path(safe).stem or "online-source"
    name = f"{stem}{guessed}"
    validate_upload_name(name)
    return name


def download_to_input(
    raw_url: str,
    input_dir: Path,
    *,
    max_bytes: int,
    kind_hint: Optional[str] = None,
) -> Tuple[Path, str, int, str]:
    """Fetch one URL into the project's input dir.

    YouTube links resolve via yt-dlp; everything else is a plain download.
    Returns (final_path, content_type, bytes_written, source_url).
    Raises URLImportError on any validation or transfer failure.
    """
    parsed = validate_source_url(raw_url)
    if is_youtube_url(parsed):
        if kind_hint == "image":
            raise URLImportError("A YouTube link is a video, not an image source")
        return _download_youtube_to_input(
            urllib.parse.urlunparse(parsed), input_dir, max_bytes=max_bytes
        )
    request = urllib.request.Request(
        urllib.parse.urlunparse(parsed),
        headers={"User-Agent": _USER_AGENT},
    )
    try:
        response = urllib.request.urlopen(request, timeout=_CONNECT_TIMEOUT)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise URLImportError(f"Could not fetch the source URL: {exc}") from exc
    with response:
        final_url = response.geturl()
        try:
            final_host = urllib.parse.urlparse(final_url).hostname or ""
            if final_host:
                _reject_non_public_host(final_host)
        except URLImportError as exc:
            raise URLImportError(f"Source redirect is not allowed: {exc}") from exc
        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type and not (
            content_type.startswith(_VIDEO_TYPES)
            or content_type.startswith(_IMAGE_TYPES)
            or content_type in _GENERIC_TYPES
            or content_type.startswith("model/")
            or content_type.startswith("application/")
        ):
            raise URLImportError(
                f"Source URL returned '{content_type or 'unknown type'}', not reference media"
            )
        try:
            filename = _filename_for_url(
                urllib.parse.urlparse(final_url),
                content_type,
                response.headers.get("Content-Disposition"),
            )
        except FileValidationError as exc:
            raise URLImportError(str(exc)) from exc
        if kind_hint:
            kind = file_kind(filename)
            if kind != kind_hint and not (
                kind_hint == "video" and kind in {"video", "image"}
            ):
                raise URLImportError(
                    f"Source URL is a {kind or 'unknown type'}, expected a {kind_hint}"
                )
        input_dir.mkdir(parents=True, exist_ok=True)
        destination = unique_destination(input_dir, filename)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        total = 0
        try:
            with temporary.open("wb") as handle:
                while True:
                    try:
                        chunk = response.read(1024 * 1024)
                    except OSError as exc:
                        raise URLImportError(f"Download interrupted: {exc}") from exc
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise URLImportError(
                            f"Online source exceeds the {max_bytes} byte limit"
                        )
                    handle.write(chunk)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        if total == 0:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise URLImportError("Online source returned an empty file")
        temporary.replace(destination)
        return destination, content_type or "application/octet-stream", total, final_url


def _download_youtube_to_input(
    page_url: str, input_dir: Path, *, max_bytes: int
) -> Tuple[Path, str, int, str]:
    """Resolve a YouTube page URL and download its video (yt-dlp).

    Single progressive file up to 1080p keeps the result a plain .mp4 the
    pipeline reads directly.  No shell is involved: the yt-dlp Python API
    runs in the caller's worker thread.
    """
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise URLImportError(
            "YouTube links need the yt-dlp package, which is not installed "
            "on the server"
        ) from exc
    try:
        parsed = validate_source_url(page_url)
    except URLImportError as exc:
        raise URLImportError(f"Rejected YouTube URL: {exc}") from exc
    if not is_youtube_url(parsed):
        raise URLImportError("Not a YouTube URL")
    input_dir.mkdir(parents=True, exist_ok=True)
    staging = unique_destination(input_dir, f"yt-{uuid.uuid4().hex}.mp4")
    temporary = staging.with_suffix(".part")
    params = {
        "format": (
            "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/"
            "b[height<=1080][ext=mp4]/b[height<=1080]/b"
        ),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "max_filesize": max_bytes,
        "outtmpl": str(temporary.with_suffix(".%(ext)s")),
        "restrictfilenames": False,
        "no_overwrites": True,
    }
    try:
        with yt_dlp.YoutubeDL(params) as ydl:
            info = ydl.extract_info(page_url, download=True)
    except DownloadError as exc:
        message = str(exc).split(":", 1)[-1].strip() or str(exc)
        raise URLImportError(_friendly_youtube_error(message)) from exc
    except Exception as exc:
        raise URLImportError(f"YouTube download failed: {exc}") from exc
    produced = sorted(
        input_dir.glob(f"{temporary.stem}.*"),
        key=lambda p: p.stat().st_size if p.is_file() else -1,
    )
    produced = [p for p in produced if p.is_file() and p.stat().st_size > 0]
    if not produced:
        raise URLImportError("YouTube download produced no video file")
    downloaded = produced[0]
    for extra in produced[1:]:
        try:
            extra.unlink()
        except OSError:
            pass
    title = str((info or {}).get("title") or "youtube-video").strip() or "youtube-video"
    video_id = str((info or {}).get("id") or "").strip()
    stem = sanitize_filename(f"{title}-{video_id}" if video_id else title,
                             fallback="youtube-video")
    candidate = f"{Path(stem).stem}{downloaded.suffix.lower()}"
    try:
        filename, kind = validate_upload_name(candidate)
    except FileValidationError as exc:
        try:
            downloaded.unlink()
        except OSError:
            pass
        raise URLImportError(
            f"YouTube delivered an unsupported format "
            f"('{downloaded.suffix or 'unknown'}'): {exc}"
        ) from exc
    if kind != "video":
        try:
            downloaded.unlink()
        except OSError:
            pass
        raise URLImportError(
            f"YouTube delivered a {kind}, not a video - pick another link"
        )
    destination = unique_destination(input_dir, filename)
    downloaded.replace(destination)
    total = destination.stat().st_size
    if total > max_bytes:
        try:
            destination.unlink()
        except OSError:
            pass
        raise URLImportError(
            f"YouTube video exceeds the {max_bytes} byte limit - "
            f"pick a shorter clip"
        )
    content_type = mimetypes.guess_type(destination.name)[0] or "video/mp4"
    return destination, content_type, total, page_url


def _friendly_youtube_error(message: str) -> str:
    lowered = message.lower()
    if "private" in lowered:
        return "That YouTube video is private - use a public link"
    if "deleted" in lowered or "unavailable" in lowered:
        return "That YouTube video is unavailable or deleted - use another link"
    if "age" in lowered and "confirm" in lowered:
        return "That YouTube video needs age confirmation - use another link"
    if "larger than max-filesize" in lowered or "file is larger" in lowered:
        return "That YouTube video is larger than the project upload limit - pick a shorter clip"
    if "sign in" in lowered or "log in" in lowered:
        return "YouTube asks for sign-in for that video - use a public link"
    if "timed out" in lowered or "timeout" in lowered:
        return "YouTube timed out - try the link again"
    return f"YouTube download failed: {message[:300]}"


def extract_url_list(meta: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
    """Pull (url, kind_hint) pairs from creation/import metadata.

    Accepts `video_url` (str), `video_urls` (list), `image_urls` (list).
    """
    pairs: List[Tuple[str, Optional[str]]] = []
    single = meta.get("video_url")
    if isinstance(single, str) and single.strip():
        pairs.append((single.strip(), "video"))
    for key, hint in (("video_urls", "video"), ("image_urls", "image")):
        value = meta.get(key)
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    pairs.append((item.strip(), hint))
    # De-duplicate while keeping order.
    seen: set[str] = set()
    unique: List[Tuple[str, Optional[str]]] = []
    for url, hint in pairs:
        if url not in seen:
            seen.add(url)
            unique.append((url, hint))
    return unique


async def import_urls_for_project(
    project: Dict[str, Any],
    file_manager: Any,
    urls: List[Tuple[str, Optional[str]]],
    *,
    max_bytes: int,
    timeout_note: str = "online source import",
) -> List[Dict[str, Any]]:
    """Download URLs into input/ and register them.  Runs blocking I/O in a thread."""
    import asyncio

    results: List[Dict[str, Any]] = []
    project_dir = file_manager.project_dir(project)
    input_dir = project_dir / "input"
    for url, hint in urls:
        destination, content_type, total, final_url = await asyncio.to_thread(
            download_to_input,
            url,
            input_dir,
            max_bytes=max_bytes,
            kind_hint=hint,
        )
        registered = file_manager.register_file(
            project,
            destination,
            original_name=destination.name,
        )
        file_manager.logger.append(
            int(project["id"]),
            project_dir,
            stage="UPLOAD",
            operation="import-url",
            status="completed",
            message=f"Imported {destination.name} ({total} bytes) from {final_url}",
        )
        results.append({**registered, "source_url": final_url})
    _ = timeout_note
    return results
