"""Entry point for the backend server.

Serves the website/ directory and exposes the API used by script.js:
  POST /api/upload  -> accept an uploaded file into uploads/
  GET  /api/jobs    -> list processing jobs
"""

import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import jobs
from processing import process_file

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / "server.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Server")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Save an uploaded file and queue it for processing."""
    dest = UPLOADS_DIR / file.filename
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    job = jobs.create(file.filename)
    process_file(job)
    logging.info("Queued job %s for %s", job["id"], file.filename)
    return JSONResponse({"job": job})


@app.get("/api/jobs")
async def list_jobs():
    return JSONResponse(jobs.list_all())


app.mount("/", StaticFiles(directory=BASE_DIR / "website", html=True), name="website")


if __name__ == "__main__":
    import uvicorn

    UPLOADS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
