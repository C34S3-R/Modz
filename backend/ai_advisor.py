"""Optional AI advisor for 3D projects (Ollama, local only).

This service never touches the reconstruction pipeline.  It reads the
already-persisted project state (validation/source-check, frame counts,
reconstruction report, mesh stats) and turns it into plain-language
guidance: which preset fits, what went wrong, what to re-shoot.

The rule-based analysis always works.  When a local Ollama server is
reachable it adds a short plain-language summary on top; when Ollama is
down the endpoint still returns the rules with ``ai_available: false``
instead of failing.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def _report(project_dir: Path) -> Dict[str, Any]:
    for candidate in (
        project_dir / "reconstruction_report.json",
        project_dir / "reconstruction" / "reconstruction_report.json",
        project_dir / "metadata" / "reconstruction_report.json",
    ):
        try:
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def _rules(
    project: Dict[str, Any],
    state: Dict[str, Any],
    report: Dict[str, Any],
) -> Dict[str, Any]:
    stages = state.get("stages", {})
    options = project.get("options") or {}
    findings: List[str] = []
    actions: List[str] = []

    validation = (stages.get("validation") or {}).get("result") or {}
    source_check = validation.get("source_check") or {}
    if source_check and not source_check.get("ok"):
        failed = [
            c.get("name")
            for c in source_check.get("checks", [])
            if isinstance(c, dict) and not c.get("ok")
        ]
        findings.append(f"Source check {source_check.get('score', '')} failed: {', '.join(failed) or 'weak input'}.")
        actions.append("Re-shoot horizontal 1080p+, slow steady orbit, object filling ~70% of the frame.")

    extraction = (stages.get("extraction") or {}).get("result") or {}
    count = extraction.get("count")
    if isinstance(count, int) and count < 60:
        findings.append(f"Only {count} frames extracted; short clips cannot cover a full orbit.")
        actions.append("Capture a longer clip (30s+) so the small preset still sees every side.")

    registered = report.get("registered_images")
    total = report.get("total_images")
    if isinstance(registered, int) and isinstance(total, int) and total:
        pct = registered / total * 100
        findings.append(f"Registered {registered}/{total} frames ({pct:.0f}%).")
        if pct < 60:
            actions.append("Keep the background static and empty; moving people/cars split the match graph.")
    points = report.get("sparse_points")
    if isinstance(points, int) and points < 5000:
        findings.append(f"Only {points} sparse points (healthy is 10k+); the mesh is mostly guesswork.")
        actions.append("Add light: overcast daylight, no harsh shadows, matte surfaces beat gloss.")

    mesh = (stages.get("mesh_validation") or {}).get("result") or {}
    if mesh.get("vertices"):
        findings.append(f"Mesh has {mesh.get('vertices')} vertices / {mesh.get('faces')} faces.")

    lights = options.get("lights")
    bussid = options.get("bussid")
    project_type = project.get("project_type") or "general"
    if project_type == "general" and bussid:
        actions.append("BUSSID export is on for a general model; switch it off unless this becomes a game mod.")
    if not lights and project_type == "vehicle":
        actions.append("Consider enabling light analysis for vehicles before export.")

    preset = project.get("preset") or "standard"
    if project_type == "general" and preset == "vehicle":
        suggestion = "small"
        actions.append("Preset 'vehicle' is heavy for a first model; 'small' (2 fps, 120 frames) is faster and enough.")
    elif project_type == "vehicle" and preset == "small":
        suggestion = "vehicle"
        actions.append("Small preset may under-sample a full vehicle; 'vehicle' preset captures more angles.")
    else:
        suggestion = preset

    if not findings:
        findings.append("No red flags in the recorded state.")
    if not actions:
        actions.append("Proceed to export; current settings look reasonable.")

    return {"findings": findings, "actions": actions, "suggestion": suggestion}


def _llm_summary(
    host: str,
    model: str,
    project: Dict[str, Any],
    rules: Dict[str, Any],
    report: Dict[str, Any],
    timeout_seconds: int = 300,
) -> Optional[str]:
    facts = {
        "name": project.get("name"),
        "type": project.get("project_type"),
        "preset": project.get("preset"),
        "status": project.get("status"),
        "findings": rules["findings"][:6],
        "registered": f"{report.get('registered_images')}/{report.get('total_images')}",
        "sparse_points": report.get("sparse_points"),
    }
    prompt = (
        "You advise a beginner doing phone-based photogrammetry. "
        "In under 120 words, explain these scan results in plain language "
        "and list the top 3 fixes, most important first. No jargon.\n"
        f"Facts: {json.dumps(facts)}"
    )
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 120}}
    ).encode()
    request = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    text = (payload.get("response") or "").strip()
    return text or None


class AIAdvisor:
    def __init__(self, project_manager: Any, config: Any):
        self.projects = project_manager
        self.config = config

    @property
    def available(self) -> bool:
        return bool(os.getenv("ADVISOR_ENABLED", "1").lower() not in {"0", "false", "no", "off"})

    def analyze(self, project_id: int) -> Dict[str, Any]:
        project = self.projects.get_or_404(project_id)
        state = self.projects.state(project_id)
        project_dir = self.projects.project_dir(project)
        report = _report(project_dir)
        rules = _rules(project, state, report)

        host = getattr(self.config, "ollama_host", "http://127.0.0.1:11434")
        model = getattr(self.config, "ollama_model", "qwen2.5-coder:1.5b")
        summary: Optional[str] = None
        ai_available = False
        if self.available:
            summary = _llm_summary(host, model, project, rules, report)
            ai_available = summary is not None

        return {
            "project_id": project_id,
            "project_type": project.get("project_type") or "general",
            "preset": project.get("preset") or "standard",
            "status": project.get("status"),
            "suggested_preset": rules["suggestion"],
            "findings": rules["findings"],
            "actions": rules["actions"],
            "ai_summary": summary,
            "ai_model": model,
            "ai_available": ai_available,
        }
