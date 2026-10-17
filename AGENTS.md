# AGENTS.md — Modz Vehicle Processing Server

## Architecture

- `website/script.js` is the executable frontend contract. Read `API_ENDPOINTS`, `Live`, and the "Backend location" block at the top of the file before changing routes or event payloads; the browser expects the project/settings/lights/export APIs, WebSocket `/ws`, and SSE `/api/events`. The UI resolves its backend at startup in this order: explicit `?api=`/saved URL, this page's origin when it is loopback, a Modz backend on the browser's own machine (`127.0.0.1`/`localhost:8002`, found through `GET /api/health`), then this page's origin. A winner on another origin is navigated to rather than called cross-origin, so everything stays same-origin; a backend that only exists on another machine is used only after an explicit consent that is remembered per origin. `API_TOKEN` is sent as `X-API-Token` (or `?token=` for downloads, SSE, and `/ws`).
- `backend/main.py` is the composition root. It creates the SQLite-backed services, starts the worker, and mounts `website/` at `/` **last**; any new API route must be registered before that catch-all mount.
- `backend/api/` contains thin routers; business logic belongs in `project_manager.py`, `file_manager.py`, `job_manager.py`, `export_service.py`, or `pipeline/`. Do not put a second synchronous pipeline in a request handler.
- `backend/pipeline/` is checkpointed and staged: video validation → extraction/filtering/preprocessing → Meshroom/reconstruction validation → Blender → light analysis/creation/validation → BUSSID prep/export/package. SQLite checkpoints preserve completed stages for retry. `backend/README.md` is the newcomer-oriented request/stage walkthrough.
- `backend/database.py` stores metadata only; inputs, intermediates, logs, and outputs stay under `projects/<slug>/`. Project rows retain their original storage root and input revision, so changing global settings or adding files cannot orphan/invalidate the wrong artifacts. `server.db`, generated project contents, and runtime directories are ignored by Git.

## Deployment context

- `/home/admin/modz` is the persistent **Matatu vehicle-processing server**, not the Ollama/Qwen AI server. Keep its process, port, database, and configuration separate; do not modify or restart Ollama/Qwen while working on this repository. A separate operator machine reaches the Matatu host through SSH; do not treat this checkout as disposable local state or move `server.db`/project workspaces to an ephemeral working directory.
- The launcher binds **localhost (`127.0.0.1`)** by default: the tool runs on whichever machine needs it and serves that machine only, so whoever starts it - you or someone on their own laptop - opens `http://localhost:<port>` locally and the pipeline runs right there. Network exposure is a deliberate, explicit opt-in: `HOST=0.0.0.0` (every interface), an explicit address, `HOST=auto` (prefer Tailscale, for a remote operator on the same tailnet), or `HOST=loopback` (same as the default). Whichever you choose, use a strong `API_TOKEN` and restrict the port with a firewall; never expose unauthenticated admin/file APIs to an untrusted network.

## Run and verify

- Install declared dependencies with `python3 -m pip install -r requirements.txt`. The current environment has no `python` alias, so use `python3` in commands.
- Start the Matatu server with `./start-matatu` (or its equivalent `./start-backend`; the launcher resolves the repository root, prefers `.venv/bin/python`, binds localhost by default, and accepts `HOST`, `PORT`, or `PYTHON_BIN`). Running `python3 backend/main.py` directly serves localhost the same way, so a fresh clone just works on the machine that runs it - no address is baked in anywhere. To serve other devices, set `HOST=0.0.0.0` or an explicit address deliberately, and set `API_TOKEN` before binding beyond loopback.
- A user systemd unit `modz-matatu.service` (enabled, `Restart=always`) runs `start-backend` on port 8002. Never run a manual server alongside it: the duplicate boots run recovery and mark the active job failed, then die on the busy port. Manage it with `systemctl --user status|restart modz-matatu`.
- Frontend-only preview: `python3 -m http.server -d website 8080`; if a backend is running on this machine the preview immediately hands the page over to it (that is the discovery order above), and with no backend anywhere the badge stays honestly offline.
- There is no repository test suite, lint, formatter, typecheck, CI, or task runner. Focused checks are `python3 -m py_compile $(find backend -name '*.py' -type f | sort)` and, when Node is installed, `node --check website/script.js`.
- External tool paths come from `.env.example`/settings (`FFMPEG_PATH`, `FFPROBE_PATH`, `MESHROOM_PATH`, `BLENDER_PATH`), not machine-specific constants. Meshroom is required for video-only photogrammetry; Blender has a logged model-copy fallback unless `REQUIRE_EXTERNAL_TOOLS=true`.

## High-risk gotchas

- Uploads are streamed in 1 MiB chunks, extension/MIME/size checked, and filenames sanitized. Never reintroduce direct `UPLOADS_DIR / user_filename` paths or shell command strings; external commands must remain argument lists. API deletion intentionally refuses original input files.
- `POST /api/projects/{id}/process` only queues a job. The single worker owns execution; it persists state before/after each stage, supports pause/cancel/retry, and must not be replaced with inline `process_file()` work. Run one Uvicorn process/worker: the live broker and job manager are in-process.
- The default concurrency gate is one because the host has limited RAM. Do not start Meshroom and Blender in parallel or delete a project's originals/intermediates during cleanup.
- `website/index.html` loads Three.js `0.160.0` from unpkg; the 3D preview needs network/CDN access even though the rest of the UI has no build step.
- API and WebSocket URLs are same-origin/relative. Keep the static mount last and keep `/ws` (not `/api/ws`) compatible with the frontend.
