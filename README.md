# Modz — Vehicle Processing Server

As part of an ongoing experiment, i created Modz, a 3d construction pipeline that converts anything from a reference video, images to a complete 3d model. It was specifically desinged to run on old hardware so it is insanely slow but the better hardware you have the better performance. It is a cpu focused pipeline created to run on an old dell latitude E6410 with a core i5 with 6 gigs of ram and that includes the entire backend system. So with better machine better results.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
./modz
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
Feel free to leave a start and fork for better improvement


