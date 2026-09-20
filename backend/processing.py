"""Processing logic for uploaded files.

Each job moves an uploaded file through the processing/ directory and
writes results to output/.
"""

import logging
import shutil
from pathlib import Path

import jobs

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
PROCESSING_DIR = BASE_DIR / "processing"
OUTPUT_DIR = BASE_DIR / "output"

logger = logging.getLogger(__name__)


def process_file(job: dict) -> None:
    """Run the processing pipeline for one job.

    Replace the placeholder work below with the real pipeline
    (e.g. rendering, transcoding, analysis).
    """
    jobs.set_status(job["id"], "processing")

    src = UPLOADS_DIR / job["filename"]
    work = PROCESSING_DIR / job["filename"]
    result = OUTPUT_DIR / job["filename"]

    try:
        if not src.exists():
            raise FileNotFoundError(src)

        PROCESSING_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)

        shutil.move(src, work)

        # TODO: actual processing step goes here.
        shutil.copy(work, result)

        work.unlink()
        jobs.set_status(job["id"], "done")
        logger.info("Job %s completed", job["id"])
    except Exception:
        jobs.set_status(job["id"], "failed")
        logger.exception("Job %s failed", job["id"])
        raise
