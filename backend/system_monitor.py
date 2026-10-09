"""Lightweight Linux server metrics without a psutil dependency."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import AppConfig


class SystemMonitor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.started_at = time.time()

    def _cpu_percent(self) -> float:
        try:
            first = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            values = [int(value) for value in first]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
            time.sleep(0.03)
            second = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            second_values = [int(value) for value in second]
            idle_delta = (second_values[3] + (second_values[4] if len(second_values) > 4 else 0)) - idle
            total_delta = sum(second_values) - total
            return round(max(0.0, min(100.0, (1 - idle_delta / total_delta) * 100)), 2) if total_delta else 0.0
        except (OSError, ValueError, IndexError):
            try:
                return round(os.getloadavg()[0], 2)
            except OSError:
                return 0.0

    def _memory(self) -> tuple[float, float]:
        values = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, raw = line.split(":", 1)
                values[key] = float(raw.strip().split()[0]) / 1024 / 1024
        except (OSError, ValueError, IndexError):
            return 0.0, 0.0
        total = values.get("MemTotal", 0.0)
        available = values.get("MemAvailable", values.get("MemFree", 0.0))
        return round(max(0.0, total - available), 2), round(total, 2)

    def _temperature(self) -> Optional[float]:
        try:
            for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
                raw = zone.read_text().strip()
                value = float(raw) / 1000 if raw.isdigit() else float(raw)
                if value > 0:
                    return round(value, 1)
        except (OSError, ValueError):
            pass
        return None

    def _processes(self) -> List[Dict[str, Any]]:
        executable = shutil.which("ps")
        if not executable:
            return []
        try:
            result = subprocess.run(
                [executable, "-eo", "pid=,comm=,%cpu=,%mem=", "--sort=-%cpu"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        processes = []
        for line in result.stdout.splitlines()[:10]:
            parts = line.split(None, 3)
            if len(parts) == 4:
                try:
                    processes.append(
                        {
                            "pid": int(parts[0]),
                            "name": parts[1],
                            "cpu_percent": float(parts[2]),
                            "memory_percent": float(parts[3]),
                        }
                    )
                except ValueError:
                    continue
        return processes

    def status(self, dependencies: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # These values are advisory diagnostics.  They are intentionally kept
        # independent of optional third-party monitoring libraries.
        cpu = self._cpu_percent()
        ram_used, ram_total = self._memory()
        disk = shutil.disk_usage(self.config.base_dir)
        result: Dict[str, Any] = {
            "online": True,
            "cpu_percent": cpu,
            "cpu": cpu,
            "ram_used_gb": ram_used,
            "ram_total_gb": ram_total,
            "ram_used": ram_used,
            "ram_total": ram_total,
            "disk_used_gb": round(disk.used / 1024**3, 2),
            "disk_total_gb": round(disk.total / 1024**3, 2),
            "disk_used": round(disk.used / 1024**3, 2),
            "disk_total": round(disk.total / 1024**3, 2),
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "temperature": self._temperature(),
            "processes": self._processes(),
            "dependencies": dependencies or self.config.tool_status(),
            "max_concurrent_jobs": self.config.max_concurrent_jobs,
        }
        return result
