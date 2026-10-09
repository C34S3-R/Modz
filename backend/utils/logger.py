"""Project-aware logging backed by both text files and SQLite."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from database import Database


class ProjectLogger:
    def __init__(self, db: Database, config: Any):
        self.db = db
        self.config = config
        self._lock = threading.RLock()
        self.server_logger = logging.getLogger("vehicle_server")

    def append(
        self,
        project_id: Optional[int],
        project_dir: Optional[Path],
        *,
        stage: str = "SYSTEM",
        operation: str = "log",
        status: str = "info",
        message: str,
        level: str = "INFO",
        external_command: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> Dict[str, Any]:
        created_at = time.time()
        timestamp = datetime.fromtimestamp(created_at).strftime("%H:%M:%S")
        line = f"[{timestamp}] [{stage.upper()}] {operation}: {message}"
        if external_command:
            line += f" | command={external_command}"
        if exit_code is not None:
            line += f" | exit={exit_code}"
        if project_dir is None and project_id is not None:
            project = self.db.fetchone(
                "SELECT slug, root_path FROM projects WHERE id = ?",
                (project_id,),
            )
            if project:
                project_dir = Path(project.get("root_path") or self.config.projects_dir) / project["slug"]
        if project_dir is not None:
            try:
                log_path = Path(project_dir) / "logs" / "pipeline.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._lock:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(line + "\n")
            except OSError:
                # Logging must never take down the request or worker.
                self.server_logger.exception("Unable to write project log %s", line)

        self.db.execute(
            """
            INSERT INTO logs
                (project_id, created_at, stage, operation, status, message, level,
                 external_command, exit_code)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                created_at,
                stage,
                operation,
                status,
                message,
                level.upper(),
                external_command,
                exit_code,
            ),
        )
        return {
            "time": created_at,
            "created_at": created_at,
            "stage": stage,
            "operation": operation,
            "status": status,
            "message": message,
            "level": level.upper(),
            "external_command": external_command,
            "exit_code": exit_code,
        }

    def project_logs(self, project_id: int, limit: int = 1000) -> List[Dict[str, Any]]:
        rows = self.db.fetchall(
            """
            SELECT created_at, stage, operation, status, message, level,
                   external_command, exit_code
            FROM logs WHERE project_id = ?
            ORDER BY created_at DESC, id DESC LIMIT ?
            """,
            (project_id, limit),
        )
        rows.reverse()
        return [
            {
                "time": row.get("created_at"),
                "created_at": row.get("created_at"),
                "stage": row.get("stage"),
                "operation": row.get("operation"),
                "status": row.get("status"),
                "message": row.get("message"),
                "level": row.get("level") or "INFO",
                "external_command": row.get("external_command"),
                "exit_code": row.get("exit_code"),
            }
            for row in rows
        ]

    def clear(self, project_id: Optional[int] = None) -> None:
        if project_id is None:
            self.db.execute("DELETE FROM logs WHERE project_id IS NULL", ())
            try:
                (self.config.logs_dir / "server.log").write_text("", encoding="utf-8")
            except OSError:
                pass
            return
        self.db.execute("DELETE FROM logs WHERE project_id = ?", (project_id,))
        project = self.db.fetchone("SELECT slug FROM projects WHERE id = ?", (project_id,))
        if not project:
            return
        # Project log files are kept as recovery artifacts; truncate the active
        # pipeline log only after the database rows have been cleared.
        path = self.config.projects_dir / project["slug"] / "logs" / "pipeline.log"
        try:
            path.write_text("", encoding="utf-8")
        except OSError:
            pass
