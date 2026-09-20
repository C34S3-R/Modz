# Modz - Vehicle Processing Server

A personal, single-user web interface (control panel) for a vehicle-processing
server running on Linux. The server handles the full workflow of turning a
reference video or images into a game-ready vehicle mod:

```
VIDEO / IMAGE / 3D MODEL
        |
        v
      UPLOAD
        |
        v
FRAME EXTRACTION
        |
        v
IMAGE PROCESSING
        |
        v
MESHROOM / ALICEVISION
        |
        v
3D MODEL PROCESSING
        |
        v
       BLENDER
        |
        v
LIGHT DETECTION / CREATION
        |
        v
    BUSSID MATERIALS
        |
        v
    BUSSID EXPORT
        |
        v
      DOWNLOAD
```

STATUS: under development. The frontend is implemented; the backend is being
worked on. The UI currently shows offline states until the backend API is
available.

## Project Structure

```
server/
|
|-- website/            Frontend (this repository section)
|   |-- index.html      App shell: sidebar, topbar, new project modal
|   |-- style.css       Dark technical theme, responsive layout
|   |-- script.js       Router, views, API layer, live updates, log viewer
|
|-- backend/            Backend (under development)
|   |-- main.py         FastAPI entry point, API endpoints
|   |-- processing.py   Processing pipeline logic
|   |-- jobs.py         Job registry and status tracking
|
|-- projects/           One folder per project (e.g. stinger/)
|-- uploads/            Incoming uploaded files
|-- processing/         Working directory for the pipeline
|-- output/             Finished exports (.bussidmod)
|-- logs/               Server and processing logs
```

## Frontend Pages

- Dashboard: server status (CPU, RAM, storage, uptime), quick actions,
  current project with live progress.
- Projects: create a project (reference video, 3D model, reference images,
  pipeline options) and browse existing ones.
- Project detail: inputs, pipeline visualization with per-stage status
  (waiting, running, done, failed), start/pause/cancel controls, project
  file browser with download and confirmed delete, error display with
  exit code, cause and recovery actions.
- Processing: overview of all active jobs.
- 3D Preview: Three.js viewer with rotate/pan/zoom, solid, wireframe and
  material modes, reset view, model statistics.
- Lights: light categories (headlights, tail lights, brake lights,
  indicators, reverse, fog, DRL), preview state presets, manual light
  editing (position, rotation, scale, material, object).
- Exports: pre-export checklist, build progress, download of the finished
  .bussidmod file.
- Logs: terminal-style log viewer with clear, download and copy.
- Settings: tool paths (Meshroom, Blender, FFmpeg), directories, maximum
  concurrent jobs (default 1, single-user server).

## Frontend / Backend Contract

The frontend never runs heavy processing. It only calls the backend:

- REST: `/api/status`, `/api/projects`, `/api/projects/{id}/process`,
  `/api/projects/{id}/status`, `/api/projects/{id}/lights`,
  `/api/projects/{id}/export`, `/api/settings`, and related endpoints.
- Live updates: WebSocket at `/ws` (SSE fallback) carrying progress,
  stage, operation, logs and errors.

All processing (frame extraction, Meshroom, Blender, OpenCV, model
conversion, BUSSID packaging) happens on the Linux server.

## Running

Serve `website/` with any static file server, or through the backend once
it is implemented (it serves the site and the API on one origin).
