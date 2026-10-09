"""Small SQLite metadata store for projects, jobs, files, lights, and logs.

Large inputs and generated artifacts stay on disk.  SQLite is intentionally
used with short-lived connections so the FastAPI request threads and the
background worker can safely share the store.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


_JSON_COLUMNS = {
    "options",
    "state",
    "error",
    "position",
    "rotation",
    "scale",
    "metadata",
    "result",
    "value",
}


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        # A short-lived connection per operation avoids sharing an SQLite
        # connection across FastAPI threads and the background worker.
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        # SQLite stores metadata and checkpoints, never the large input files.
        # IF NOT EXISTS makes startup safe for a fresh or existing checkout.
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'idle',
                    current_stage TEXT,
                    progress REAL NOT NULL DEFAULT 0,
                    options TEXT NOT NULL DEFAULT '{}',
                    state TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    root_path TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL DEFAULT 1,
                    project_type TEXT NOT NULL DEFAULT 'vehicle',
                    preset TEXT NOT NULL DEFAULT 'standard'
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    type TEXT NOT NULL DEFAULT 'pipeline',
                    status TEXT NOT NULL DEFAULT 'queued',
                    current_stage TEXT,
                    operation TEXT,
                    progress REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    original_name TEXT,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    mime_type TEXT,
                    size INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(project_id, path)
                );

                CREATE TABLE IF NOT EXISTS lights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    position TEXT NOT NULL DEFAULT '[0, 0, 0]',
                    rotation TEXT NOT NULL DEFAULT '[0, 0, 0]',
                    scale TEXT NOT NULL DEFAULT '[1, 1, 1]',
                    material TEXT NOT NULL DEFAULT 'LightWhite',
                    object_name TEXT,
                    status TEXT NOT NULL DEFAULT 'Detected',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(project_id, name)
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    created_at REAL NOT NULL,
                    stage TEXT,
                    operation TEXT,
                    status TEXT,
                    message TEXT NOT NULL,
                    level TEXT NOT NULL DEFAULT 'INFO',
                    external_command TEXT,
                    exit_code INTEGER
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (project_id, stage)
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_pipeline
                    ON jobs(project_id) WHERE type = 'pipeline' AND status IN ('queued', 'running', 'paused', 'cancelling');
                CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_export
                    ON jobs(project_id) WHERE type = 'export' AND status IN ('queued', 'running', 'paused', 'cancelling');
                CREATE INDEX IF NOT EXISTS idx_files_project ON files(project_id);
                CREATE INDEX IF NOT EXISTS idx_logs_project ON logs(project_id, created_at);
                """
            )
            project_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(projects)")
            }
            if "root_path" not in project_columns:
                connection.execute(
                    "ALTER TABLE projects ADD COLUMN root_path TEXT NOT NULL DEFAULT ''"
                )
            if "revision" not in project_columns:
                connection.execute(
                    "ALTER TABLE projects ADD COLUMN revision INTEGER NOT NULL DEFAULT 1"
                )
            if "project_type" not in project_columns:
                connection.execute(
                    "ALTER TABLE projects ADD COLUMN project_type TEXT NOT NULL DEFAULT 'vehicle'"
                )
            if "preset" not in project_columns:
                connection.execute(
                    "ALTER TABLE projects ADD COLUMN preset TEXT NOT NULL DEFAULT 'standard'"
                )
            file_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(files)")
            }
            if "revision" not in file_columns:
                connection.execute(
                    "ALTER TABLE files ADD COLUMN revision INTEGER NOT NULL DEFAULT 1"
                )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _decode(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        value = dict(row)
        for key in _JSON_COLUMNS & value.keys():
            raw = value[key]
            if raw is None or raw == "":
                value[key] = None
                continue
            try:
                value[key] = json.loads(raw)
            except (TypeError, ValueError):
                value[key] = raw
        return value

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._write_lock:
            connection = self._connect()
            try:
                with connection:
                    cursor = connection.execute(sql, params)
                return int(cursor.lastrowid or cursor.rowcount)
            finally:
                connection.close()

    def executemany(self, sql: str, params: Iterable[Sequence[Any]]) -> None:
        with self._write_lock:
            connection = self._connect()
            try:
                with connection:
                    connection.executemany(sql, params)
            finally:
                connection.close()

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        connection = self._connect()
        try:
            return self._decode(connection.execute(sql, params).fetchone())
        finally:
            connection.close()

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        connection = self._connect()
        try:
            return [self._decode(row) for row in connection.execute(sql, params).fetchall()]
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run several writes atomically."""

        with self._write_lock:
            connection = self._connect()
            try:
                with connection:
                    yield connection
            finally:
                connection.close()

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        row = self.fetchone(sql, params)
        if not row:
            return default
        return next(iter(row.values()), default)
