"""Backend package for the vehicle processing server.

The application supports both ``backend.main:app`` from the repository root and
``main:app`` with ``backend/`` on ``PYTHONPATH``.  Exposing the backend
directory here keeps the internal module imports consistent for both forms.
"""

from pathlib import Path
import sys

_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
