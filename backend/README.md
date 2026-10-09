# Backend onboarding guide

This directory contains the server-side application. The browser in
`website/` never performs vehicle processing; it only sends HTTP/WebSocket
requests and renders the state returned by this code.

## Request lifecycle

1. `main.py` creates the configuration, SQLite database, managers, and FastAPI
   application.
2. A router in `api/` validates the HTTP request and calls a manager.
3. `project_manager.py` and `file_manager.py` store project metadata and stream
   uploads into a project directory.
4. `job_manager.py` puts a job in SQLite and a background worker runs it.
5. `pipeline/runner.py` reads and writes checkpoints around each stage.
6. Stage modules in `pipeline/` do the actual validation, media work, model
   preparation, light metadata work, and packaging.
7. `realtime.py` sends progress events to WebSocket/SSE clients while
   `project_manager.status()` remains the authoritative snapshot.

A request handler should stay short. Expensive work belongs in the worker, not
in an `async def` route handler.

## Important directories

- `config.py` reads environment variables and defines safe defaults.
- `database.py` stores small metadata records. Large files stay on disk.
- `api/` contains thin HTTP adapters; it should not contain processing logic.
- `pipeline/` contains stage implementations and the checkpoint runner.
- `reconstruction/` is the standalone CPU-only reconstruction engine (full
  plan: video/photos → sparse → dense → mesh → low-poly → texture →
  `.bussidmod`, checkpointed in `reconstruction/project_state.json`) with
  its own CLI; see `reconstruction/README.md`. It is wired into the web
  pipeline: the `meshroom` stage runs it through the texture stage and
  bridges the model into `reconstruction/` for the stages that follow
  (`pipeline/meshroom.py`).
- `utils/` contains reusable security, logging, filesystem, and subprocess
  helpers.
- `blender_scripts/` contains code executed *inside Blender*, not by FastAPI.
- `workers/` contains compatibility exports for the worker implementation.

## Files and recovery

Each project has a directory such as:

```text
projects/my_project/
  input/             originals; never overwrite these
  metadata/          ffprobe and validation metadata
  frames/            extracted/copied frame inputs
  processed_frames/  optional preprocessing output
  reconstruction/    photogrammetry output
  blender/           prepared model output
  lights/            light analysis/configuration manifests
  bussid/            BUSSID preparation output
  exports/           final .bussidmod package
  logs/              project and external-tool logs
  temp/              transient work
```

`project_state.json` mirrors the SQLite state for easy inspection. SQLite is
still authoritative. Adding an input or changing processing options increments
the project's input revision and invalidates dependent checkpoints; old files
remain available for recovery but are not selected for the new run.

## Job states

`queued → running → completed` is the normal path. `paused`, `cancelling`,
`cancelled`, and `failed` are alternate terminal/intermediate states. Only the
worker changes a running job. API routes request a transition; they do not
execute a pipeline stage themselves.

## External tools

Commands are launched with argument arrays and are never built as shell
strings. The configured FFmpeg, FFprobe, Meshroom, and Blender paths are
checked at runtime. Missing tools become stage errors with log/exit-code
information; they should not crash the FastAPI process.

## Safe changes

- Preserve the final `app.mount("/", ...)` in `main.py`; API routes registered
  after the static mount can be shadowed by it.
- Keep uploads streamed and path-checked.
- Do not delete `input/` files during normal cleanup.
- Run `python3 -m py_compile $(find backend -name '*.py' -type f | sort)` after
  backend edits and use `./start-backend` for a local/server smoke test.
