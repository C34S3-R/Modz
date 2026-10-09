# Matatu — Vehicle Processing Server

This repository is the **Matatu vehicle-processing server**. It is separate from
any Ollama/Qwen AI server and must use its own process, port, database, and
configuration.

A personal, single-user control panel for turning a reference video, images, or
an existing 3D model into a BUSSID-oriented vehicle package. The browser is
only the interface; FastAPI owns uploads, metadata, processing jobs, and files.

## Architecture

- `website/` — no-build HTML/CSS/vanilla ES-module frontend. `script.js` is the
  executable API contract consumed by the UI.
- `backend/main.py` — FastAPI composition root. It starts the worker, exposes
  the API, and mounts the website last at `/` so the UI and API share an origin.
- `backend/api/` — thin HTTP routers for projects, files, processing, lights,
  exports, and system endpoints.
- `backend/database.py` — SQLite metadata store. Originals and generated
  artifacts remain on disk; the database stores jobs, files, checkpoints,
  lights, settings, and log records.
- `backend/job_manager.py` — one background worker and shared resource gate.
  Jobs are queued, pausable, cancellable, retryable, and recoverable after a
  restart. A SQLite `running` job is marked failed on restart; queued jobs are
  picked up again.
- `backend/pipeline/` — isolated stage modules for video validation, frame
  extraction/filtering/preprocessing, the Meshroom stage (which runs the
  CPU-only reconstruction engine by default — see
  `backend/reconstruction/README.md`; `MESHROOM_BACKEND=external` restores
  the external executable), reconstruction validation, Blender, lights, BUSSID
  preparation, and packaging.

Runtime directories are kept in the repository but their contents are ignored:

```
uploads/                 legacy upload area
processing/              shared temporary work
output/                  server-level output copies
projects/<project>/      input and all project artifacts
  input/ metadata/ frames/ processed_frames/ rejected_frames/
  reconstruction/ blender/ lights/ bussid/ exports/ logs/ temp/
logs/                    server.log
server.db                SQLite metadata
```

## Setup and running

Python 3.10+ and the packages in `requirements.txt` are required:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

After installing dependencies, start the combined API/UI server with one command from any directory:

```bash
./start-matatu
```

`start-matatu` binds a LAN/Tailscale address automatically (see the startup log for the exact URL). Override it with `HOST`, `PORT`, `PYTHON_BIN`, or extra Uvicorn arguments. The equivalent explicit command is:

```bash
PYTHONPATH=backend python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

The equivalent backend-directory command is:

```bash
cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

For a frontend-only preview, use a separate port. API and live updates will be
unavailable, so the UI intentionally displays offline state:

```bash
python3 -m http.server -d website 8080
```

Copy `.env.example` to the server environment (the app does not load `.env`
files automatically) and adjust paths for the Dell host. Important variables
include `FFMPEG_PATH`, `FFPROBE_PATH`, `MESHROOM_PATH`, `BLENDER_PATH`,
`MESHROOM_BACKEND` (default engine, `external` for a real Meshroom install),
`MAX_CONCURRENT_JOBS`, `MAX_UPLOAD_BYTES`, `MAX_PROJECT_STORAGE_BYTES`,
`MAX_PROJECT_FILES`, `MAX_EXPORT_BYTES`, and `REQUIRE_EXTERNAL_TOOLS`. The
launcher selects a non-loopback interface for this server. Set a long
`API_TOKEN` and firewall the port before exposing it to an untrusted network;
API clients must then send `X-API-Token` or `Authorization: Bearer ...` (the
browser UI needs a matching proxy/session setup).

## API contract

The frontend calls these relative paths:

- `GET /api/status` and `GET /api/system/status` — online state, CPU, RAM,
  disk, uptime, temperature, processes, and external-tool availability.
- `GET/POST /api/projects`, `GET/DELETE /api/projects/{id}` — project lifecycle.
  `POST /api/projects` accepts either JSON metadata or the frontend's multipart
  `meta` plus repeated `files` fields.
- `GET /api/projects/{id}/status`, `/logs`, `/checkpoints` — state, ordered log
  entries, and persisted stage checkpoints.
- `POST /api/projects/{id}/process`, `/pause`, `/resume`, `/cancel`, `/retry` —
  background job control. Processing responses return immediately with a job
  ID; the worker performs the work.
- `GET /api/projects/{id}/files`, `POST /api/projects/{id}/upload` (also accepted at `/files`), and
  `GET /api/projects/{id}/files/download?path=...` — project file operations.
  Original input files are protected from API deletion.
- `GET/PUT/POST/DELETE /api/projects/{id}/lights` and
  `GET /api/projects/{id}/lights/analysis` — light metadata and analysis.
- `GET /api/projects/{id}/model` — model URL and basic mesh statistics for the
  Three.js preview.
- `GET /api/projects/{id}/export/checklist`,
  `POST /api/projects/{id}/export`, `POST /api/projects/{id}/export/cancel`, and
  `GET /api/projects/{id}/export/status` — asynchronous package build.
- `GET/PUT /api/settings`, `GET/DELETE /api/logs` — persisted settings and
  server logs.
- `WS /ws` with `GET /api/events` SSE fallback — progress, stage, operation,
  log, and error events.

The older `POST /api/upload` and `GET /api/jobs` routes remain as compatibility
wrappers around projects and the SQLite job manager.

## Pipeline behavior

Every project stores stage state in `projects/<slug>/` and the `state` column.
Completed stages are checkpoints, so retry/resume does not redo successful
work. Originals are never overwritten: frame extraction, preprocessing,
Blender output, BUSSID assets, and exports are separate directories.

External tools are invoked with argument arrays and captured logs:

- FFprobe/FFmpeg validate and extract video when a video is present.
- The photogrammetry stage named `Meshroom` runs the CPU-only reconstruction
  engine by default (video/photos → sparse → dense → mesh → low-poly →
  textured model; no CUDA needed). With `MESHROOM_BACKEND=external` it
  invokes the configured Meshroom/AliceVision executable instead, where a
  missing or failed binary becomes a failed stage with exit code/details
  rather than a FastAPI crash.
- Blender runs `backend/blender_scripts/prepare_model.py` when configured. If
  Blender is absent, model-only jobs use a clearly logged compatibility copy so
  the rest of the recoverable pipeline can be exercised; set
  `REQUIRE_EXTERNAL_TOOLS=true` to make that fallback an error.
- Light analysis is deliberately conservative and model/reference-name based;
  it records detected objects and materials in the project manifest, while a
  future vision/Blender analyzer can replace that adapter without changing the
  API.
- The exporter creates a `.bussidmod` ZIP containing an explicit current-revision
  asset allowlist under the project's `exports/`, and mirrors it to `output/`.

Only one heavy pipeline worker runs at a time by default. The worker also
checks available disk/RAM before starting and keeps the original inputs,
intermediate results, logs, and final outputs separate.

## Verification

There is no repository test suite or lint configuration yet. Useful focused
checks are:

```bash
python3 -m py_compile $(find backend -name '*.py' -type f | sort)
node --check website/script.js       # if Node.js is installed
```

For an end-to-end smoke check, start the server and exercise `GET /api/status`,
create a project through the UI, then watch `GET /api/projects/{id}/status` or
connect to `WS /ws`.
