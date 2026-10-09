"""Zero-cost online image search for reference photos.

Two keyless sources (stdlib urllib, no API keys, no signup):

* Openverse (api.openverse.org) - CC-licensed images aggregated from
  Flickr and others; anonymous requests are rate-limited but free.
* Wikimedia Commons (commons.wikimedia.org/w/api.php) - free media
  repository, generous limits, direct upload.wikimedia.org file URLs.

Results carry source-page and license fields so the UI can credit the
photographer.  Downloading a chosen result reuses the guarded importer
in utils/url_import.py (public-host check, type and size caps).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List


class OnlineSearchError(ValueError):
    pass


_USER_AGENT = "ModzVehicleServer/1.0 (reference-media search)"
_TIMEOUT = 20


def _get_json(url: str, params: Dict[str, Any]) -> Any:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": _USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise OnlineSearchError(f"Search provider unreachable: {exc}") from exc


def search_openverse(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search Openverse images (keyless)."""
    payload = _get_json(
        "https://api.openverse.org/v1/images/",
        {
            "q": query,
            "page_size": max(1, min(30, limit)),
            "fields": "id,title,url,foreign_landing_url,width,height,license,provider",
        },
    )
    results = []
    for item in (payload.get("results") or []):
        direct = (item.get("url") or "").strip()
        if not direct:
            continue
        results.append(
            {
                "provider": "openverse",
                "source": item.get("provider") or "openverse",
                "title": item.get("title") or "Untitled",
                "image_url": direct,
                "page_url": item.get("foreign_landing_url") or "",
                "width": item.get("width"),
                "height": item.get("height"),
                "license": item.get("license") or "unknown",
            }
        )
    return results


def search_commons(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search Wikimedia Commons files (keyless)."""
    payload = _get_json(
        "https://commons.wikimedia.org/w/api.php",
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,
            "gsrlimit": max(1, min(30, limit)),
            "prop": "imageinfo",
            "iiprop": "url|size",
        },
    )
    results = []
    pages = (payload.get("query") or {}).get("pages") or {}
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        direct = (info.get("url") or "").split("?")[0]
        if not direct:
            continue
        title = (page.get("title") or "").removeprefix("File:")
        thumb = (
            "https://commons.wikimedia.org/w/index.php?title=Special:FilePath"
            f"&file={urllib.parse.quote(title)}&width=320"
        )
        results.append(
            {
                "provider": "commons",
                "source": "Wikimedia Commons",
                "title": title or "Untitled",
                "image_url": direct,
                "thumbnail_url": thumb,
                "page_url": info.get("descriptionurl") or "",
                "width": info.get("width"),
                "height": info.get("height"),
                "license": "see Commons page",
            }
        )
    return results


def search_images(
    query: str, *, source: str = "all", limit: int = 20
) -> Dict[str, Any]:
    """Search one or all providers.  Never raises for empty results."""
    query = (query or "").strip()
    if not query:
        raise OnlineSearchError("A search term is required")
    if len(query) > 200:
        raise OnlineSearchError("Search term is too long")
    per = max(1, min(30, limit))
    providers = {"openverse": search_openverse, "commons": search_commons}
    if source != "all" and source not in providers:
        raise OnlineSearchError("Unknown source (use openverse, commons, or all)")
    wanted = [source] if source != "all" else ["openverse", "commons"]
    results: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    for name in wanted:
        try:
            results.extend(providers[name](query, per))
        except OnlineSearchError as exc:
            errors[name] = str(exc)
    return {"query": query, "count": len(results), "results": results, "errors": errors}
