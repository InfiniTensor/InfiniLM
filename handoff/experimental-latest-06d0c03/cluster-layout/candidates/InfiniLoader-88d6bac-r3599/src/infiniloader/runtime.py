"""Local process ownership, atomic metadata, and advisory locks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import tempfile
import time
from typing import Any


def validate_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name) or ".." in name:
        raise ValueError("name must be 1-96 ASCII letters, digits, '.', '_' or '-', without '..'")
    return name


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class FileLock:
    """Never unlink lock files: waiters must keep locking the same inode."""

    def __init__(self, path: Path, timeout: float = 30.0, shared: bool = False):
        self.path, self.timeout, self.shared = path, timeout, shared
        self.fd: int | None = None

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        deadline = time.monotonic() + self.timeout
        operation = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
        try:
            while True:
                try:
                    fcntl.flock(self.fd, operation | fcntl.LOCK_NB)
                    return self
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"another operation owns {self.path}")
                    time.sleep(0.05)
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def __exit__(self, *_):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def process_token(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"{boot}:{fields[19]}"
    except (OSError, IndexError, ValueError):
        return None


def process_matches(pid: int, token: str | None) -> bool:
    return token is not None and process_token(pid) == token


def signal_owned_process(pid: int, token: str | None) -> None:
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        fd = os.pidfd_open(pid)
        try:
            if not process_matches(pid, token):
                raise RuntimeError("refusing to signal a stale or unverified PID")
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        finally:
            os.close(fd)
    else:
        if not process_matches(pid, token):
            raise RuntimeError("refusing to signal a stale or unverified PID")
        os.kill(pid, signal.SIGTERM)
