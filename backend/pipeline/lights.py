"""Reference/model light analysis and metadata creation."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .common import PipelineContext, StageError


def analyze(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.check()
    model = ctx.model_file()
    model_path = model[1] if model else None
    from light_manager import LightManager

    analysis = LightManager(ctx.db, ctx.logger).analysis(ctx.project, model_path)
    ctx.results["light_analysis"] = analysis
    analysis_path = ctx.write_json("lights", "analysis.json", analysis)
    ctx.register(analysis_path, "metadata")
    ctx.log(
        "Light analysis completed: " + (", ".join(analysis["lights"]) or "no named lights detected"),
        stage="LIGHTS",
        operation="analyze",
        status="completed",
    )
    return analysis


def create(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.check()
    analysis = ctx.results.get("light_analysis") or {}
    from light_manager import LightManager

    manager = LightManager(ctx.db, ctx.logger)
    created = manager.create_detected(ctx.project_id, analysis)
    result = {"count": len(created), "lights": [item["name"] for item in created]}
    manifest_path = ctx.write_json("lights", "manifest.json", {"lights": created})
    ctx.register(manifest_path, "metadata")
    ctx.log(
        f"Created {len(created)} light metadata entries",
        stage="LIGHTS",
        operation="create",
        status="completed",
    )
    return result


def validate(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.check()
    from light_manager import LightManager

    lights = LightManager(ctx.db, ctx.logger).list(ctx.project_id)
    names = [item["name"] for item in lights]
    if len(names) != len(set(names)):
        raise StageError("LIGHT VALIDATION FAILED: duplicate light names were detected")
    invalid = [
        item["name"]
        for item in lights
        if not item.get("name") or len(item.get("position", [])) != 3
    ]
    if invalid:
        raise StageError(
            "LIGHT VALIDATION FAILED: one or more lights have invalid transforms",
            details={"lights": invalid},
        )
    result = {"count": len(lights), "valid": True}
    if not lights:
        result["warning"] = "No light objects were detected; manual configuration is required"
    ctx.log(
        f"Validated {len(lights)} light entries",
        stage="LIGHT_VALIDATION",
        operation="validate",
        status="completed",
    )
    return result
