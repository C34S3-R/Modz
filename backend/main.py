"""FastAPI entry point for the vehicle processing server.

The application deliberately keeps the HTTP layer thin: metadata is stored in
SQLite, large files live under each project's directory, and a single background
worker executes the checkpointed pipeline.
"""

from __future__ import annotations

import hmac
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

# The documented development command runs ``main:app`` with backend/ as the
# working directory.  Add the directory explicitly as well so importing
# ``backend.main`` from the repository root has identical module behavior.
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.advisor import router as advisor_router
from api.exports import router as exports_router
from api.files import router as files_router
from api.lights import router as lights_router
from api.processing import router as processing_router
from api.processing import ws_router
from api.projects import router as projects_router
from api.search import router as search_router
from api.system import router as system_router
from ai_advisor import AIAdvisor
from config import AppConfig
from database import Database
from export_service import ExportService
from file_manager import FileManager
from job_manager import JobManager
from light_manager import LightManager
from pipeline.runner import PipelineRunner
from project_manager import ProjectManager
from realtime import EventBroker
from settings_manager import SettingsManager
from system_monitor import SystemMonitor
from utils.logger import ProjectLogger


@dataclass
class Services:
    config: AppConfig
    db: Database
    logger: ProjectLogger
    broker: EventBroker
    settings: SettingsManager
    files: FileManager
    projects: ProjectManager
    lights: LightManager
    system: SystemMonitor
    jobs: JobManager
    exports: ExportService
    advisor: AIAdvisor


def build_services() -> Services:
    # Build dependencies once and share them between HTTP requests and worker
    # threads.  The objects are intentionally small services rather than
    # module-level globals scattered across route handlers.
    config = AppConfig.from_env()
    config.ensure_directories()
    logging.basicConfig(
        filename=config.logs_dir / "server.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    db = Database(config.database_path)
    logger = ProjectLogger(db, config)
    broker = EventBroker()
    settings = SettingsManager(db, config)
    files = FileManager(db, config, logger)
    projects = ProjectManager(db, config, logger, files)
    lights = LightManager(db, logger)
    system = SystemMonitor(config)
    jobs = JobManager(db, projects, PipelineRunner(projects, db, config, settings, logger), config, logger, broker)
    exports = ExportService(db, projects, config, settings, logger, broker, jobs.resource_gate)
    advisor = AIAdvisor(projects, config)
    return Services(
        config=config,
        db=db,
        logger=logger,
        broker=broker,
        settings=settings,
        files=files,
        projects=projects,
        lights=lights,
        system=system,
        jobs=jobs,
        exports=exports,
        advisor=advisor,
    )


services = build_services()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Startup/shutdown hooks are the right place for process-wide resources.
    # FastAPI handles HTTP requests; JobManager owns the long-running worker.
    services.jobs.start()
    missing = [
        name
        for name, status in services.config.tool_status().items()
        if not status["available"]
    ]
    if missing:
        services.logger.server_logger.warning(
            "Optional external tools unavailable: %s; compatible fallbacks will be used where safe",
            ", ".join(missing),
        )
    services.logger.server_logger.info("Vehicle processing backend ready")
    try:
        yield
    finally:
        services.jobs.stop()
        services.exports.close()


app = FastAPI(title="Vehicle Processing Server", lifespan=lifespan)
app.state.services = services


def _provided_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.headers.get("x-api-token")


@app.middleware("http")
async def optional_api_token(request: Request, call_next):
    # API_TOKEN is optional for a private loopback deployment and required by
    # convention when the launcher exposes a LAN/Tailscale interface.
    token = services.config.api_token
    if (
        token
        and request.method != "OPTIONS"
        and (request.url.path.startswith("/api/") or request.url.path == "/api")
    ):
        provided = _provided_token(request)
        if not provided or not hmac.compare_digest(provided, token):
            return JSONResponse({"detail": "API token required"}, status_code=401)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(services.config.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes must be registered before the catch-all static mount.  The mount
# serves the browser from the same origin, which keeps relative /api calls and
# WebSocket connections simple.
app.include_router(system_router)
app.include_router(projects_router)
app.include_router(files_router)
app.include_router(processing_router)
app.include_router(lights_router)
app.include_router(search_router)
app.include_router(exports_router)
app.include_router(advisor_router)
app.include_router(ws_router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "vehicle-processing"}


# Keep the website and API on one origin.  This must remain the final route.
website_dir = services.config.base_dir / "website"
if not website_dir.is_dir():
    website_dir = BACKEND_DIR.parent / "website"
app.mount("/", StaticFiles(directory=website_dir, html=True), name="website")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8002")))
