"""Safe subprocess execution with cancellation and log capture."""

from __future__ import annotations

import os
import select
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Optional, Sequence


class ProcessCancelled(RuntimeError):
    pass


class CommandError(RuntimeError):
    def __init__(self, command: Sequence[str], returncode: int, output: str = ""):
        self.command = list(command)
        self.returncode = returncode
        self.output = output
        rendered = " ".join(shlex.quote(part) for part in command)
        super().__init__(f"Command failed with exit code {returncode}: {rendered}")


def _tail(path: Path, limit: int = 6000) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _terminate_process(process: subprocess.Popen, force: bool = False) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        else:
            process.kill() if force else process.terminate()
    except (OSError, ProcessLookupError):
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError):
            pass
        process.wait()


def run_capture_command(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    log_path: Path,
    control: Any = None,
    timeout_seconds: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """Run a short command and capture stdout while retaining cancellation/logging."""

    rendered = " ".join(shlex.quote(str(part)) for part in command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
    except FileNotFoundError as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {rendered}\nexit=127\n")
        raise CommandError(command, 127, "executable not found") from exc
    chunks: list[bytes] = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n")
        while process.poll() is None:
            if control is not None:
                try:
                    control.raise_if_cancelled()
                except ProcessCancelled:
                    _terminate_process(process)
                    raise
            if timeout_seconds and time.monotonic() - started > timeout_seconds:
                _terminate_process(process, force=True)
                raise CommandError(command, 124, _tail(log_path))
            if process.stdout is not None:
                readable, _, _ = select.select([process.stdout], [], [], 0.1)
                if readable:
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if chunk:
                        chunks.append(chunk)
                        log.write(chunk.decode(errors="replace"))
                        log.flush()
        if process.stdout is not None:
            remaining = process.stdout.read()
            if remaining:
                chunks.append(remaining)
                log.write(remaining.decode(errors="replace") if isinstance(remaining, bytes) else remaining)
        return_code = process.wait()
        log.write(f"exit={return_code} elapsed={time.monotonic() - started:.2f}s\n")
    output = b"".join(chunks).decode(errors="replace")
    if return_code != 0:
        raise CommandError(command, return_code, output)
    return subprocess.CompletedProcess(command, return_code, output, "")


def run_command(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    log_path: Optional[Path] = None,
    control: Any = None,
    timeout_seconds: Optional[int] = None,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run an argument list without a shell and stream output to a log file.

    The caller supplies the executable and arguments separately, so filenames
    from uploads cannot become shell syntax.
    """

    if not command:
        raise ValueError("command cannot be empty")
    if log_path is None:
        raise ValueError("log_path is required so external output is retained")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(shlex.quote(str(part)) for part in command)
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
    except FileNotFoundError as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {rendered}\nexit=127\n")
        raise CommandError(command, 127, "executable not found") from exc

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n")
        log.flush()
        stream = process.stdout
        while process.poll() is None:
            if control is not None:
                try:
                    control.raise_if_cancelled()
                except ProcessCancelled:
                    _terminate_process(process)
                    raise
            if timeout_seconds and time.monotonic() - started > timeout_seconds:
                _terminate_process(process, force=True)
                raise CommandError(command, 124, _tail(log_path))
            if stream is not None:
                try:
                    readable, _, _ = select.select([stream], [], [], 0.1)
                except (OSError, ValueError):
                    readable = []
                if readable:
                    chunk = os.read(stream.fileno(), 65536)
                    if chunk:
                        log.write(chunk.decode(errors="replace"))
                        log.flush()
                    else:
                        stream.close()
                        stream = None
            else:
                time.sleep(0.05)
        if stream is not None:
            remaining = stream.read()
            if remaining:
                log.write(remaining.decode(errors="replace") if isinstance(remaining, bytes) else remaining)
        return_code = process.wait()
        elapsed = time.monotonic() - started
        log.write(f"exit={return_code} elapsed={elapsed:.2f}s\n")
    if return_code != 0:
        raise CommandError(command, return_code, _tail(log_path))
    return subprocess.CompletedProcess(command, return_code, "", "")
