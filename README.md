# Modz — Vehicle Processing Server

Turn a reference video, photos, or an existing 3D model into a BUSSID-ready vehicle package. Web UI + FastAPI backend + single background worker. CPU-only photogrammetry, no GPU needed.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
./start-matatu
```

Serves the UI and API on port 8002. Set `API_TOKEN` and firewall the port before exposing it beyond localhost.

## Layout

- `website/` — frontend (no build step)
- `backend/main.py` — API entry point
- `backend/pipeline/` — processing stages
- `backend/reconstruction/` — photogrammetry engine (see its README)
- `projects/` — per-project inputs and outputs (ignored by git)
- `server.db` — metadata (ignored by git)

## Check

```bash
python3 -m py_compile $(find backend -name '*.py' -type f | sort)
node --check website/script.js
```
