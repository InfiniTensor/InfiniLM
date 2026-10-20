from __future__ import annotations

import json
import os
import signal
import socket
import time
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import struct
import threading
import uuid
import stat
from pathlib import Path
from typing import Any

from .cache import CachedModel, build_view, human_bytes, load_model_into_memory, source_identity, _safe_remove_view, collect_source_files
from .runtime import FileLock, atomic_json, atomic_text, process_token, process_matches, read_json
from .local_cache import lease_path, verify_cache
from .memory import ResidentReservation


class WeightLoaderDaemon:
    def __init__(
        self,
        *,
        name: str,
        source: Path,
        view: Path,
        socket_path: Path,
        pid_path: Path,
        state_path: Path,
        upstream_source: Path | None,
        local_cache_path: Path | None,
        include_patterns: tuple[str, ...],
        exclude_patterns: tuple[str, ...],
        replace: bool,
        force: bool,
        max_memory_fraction: float,
        chunk_size: int,
        workers: int,
        quiet: bool = False,
        identity: dict | None = None,
        max_memory_bytes: int | None = None,
        reserve_bytes: int = 256 * 1024 * 1024,
        max_resident_bytes: int | None = None,
        idle_ttl: float | None = None,
    ) -> None:
        self.name = name
        self.source = source
        self.view = view
        self.socket_path = socket_path
        self.pid_path = pid_path
        self.state_path = state_path
        self.upstream_source = upstream_source
        self.local_cache_path = local_cache_path
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.replace = replace
        self.force = force
        self.max_memory_fraction = max_memory_fraction
        self.chunk_size = chunk_size
        self.workers = workers
        self.quiet = quiet
        self.identity = identity or {}
        self.process_token = process_token(os.getpid())
        self._owns_runtime = False
        self.max_memory_bytes, self.reserve_bytes = max_memory_bytes, reserve_bytes
        self.max_resident_bytes = max_resident_bytes
        self._reservation = None
        self.idle_ttl = idle_ttl
        self._management_lock = threading.RLock()
        self._leases: dict[str, dict] = {}
        self.last_used_at = time.time()
        self._last_use = time.monotonic()
        self.reuse_requests = 0
        self.cached: CachedModel | None = None
        self.started_at = time.time()
        self.load_started_at: float | None = None
        self.load_finished_at: float | None = None
        self._stop = False
        self._remove_view_on_exit = False
        self._connections = threading.BoundedSemaphore(16)
        self._claimed_runtime: dict[Path, tuple[int, int]] = {}
        self._socket_identity: dict | None = None

    def log(self, message: str) -> None:
        if not self.quiet:
            print(message, flush=True)

    def _write_state(self, state: dict[str, Any]) -> None:
        state.setdefault("name", self.name)
        state.setdefault("runtime_format", 1)
        state.setdefault("pid", os.getpid())
        state.setdefault("source", str(self.source))
        state.setdefault("view", str(self.view))
        state.setdefault("socket", str(self.socket_path))
        state.setdefault("started_at", self.started_at)
        state.setdefault("identity", self.identity)
        state.setdefault("process_token", self.process_token)
        state.setdefault("idle_ttl", self.idle_ttl)
        state.setdefault("memory_budget", self._reservation.info if self._reservation else None)
        state.setdefault("runtime_socket", self._socket_identity)
        atomic_json(self.state_path, state)
        self._claim_runtime_file(self.state_path)

    def _progress(self, copied: int, total: int, rel: Path) -> None:
        pct = copied / total * 100 if total else 100.0
        elapsed = time.time() - self.load_started_at if self.load_started_at else 0.0
        rate = copied / elapsed if elapsed > 0 and copied > 0 else None
        remaining = total - copied
        eta = remaining / rate if rate and remaining > 0 else None
        self._write_state(
            {
                "state": "loading",
                "load_started_at": self.load_started_at,
                "copied_bytes": copied,
                "copied_human": human_bytes(copied),
                "total_bytes": total,
                "total_human": human_bytes(total),
                "progress": copied / total if total else 1.0,
                "current_file": None if rel.as_posix() == "." else rel.as_posix(),
                "elapsed_seconds": elapsed,
                "rate_bytes_per_second": rate,
                "rate_human_per_second": f"{human_bytes(int(rate))}/s" if rate else None,
                "eta_seconds": eta,
            }
        )
        self.log(f"[load] {pct:6.2f}% {human_bytes(copied)} / {human_bytes(total)} cached; last={rel.as_posix()}")

    def load(self) -> None:
        self.log(f"[load] source={self.source}")
        self.load_started_at = time.time()
        self._write_state({"state": "loading", "load_started_at": self.load_started_at, "copied_bytes": 0})
        self.cached = load_model_into_memory(
            self.source,
            include_patterns=self.include_patterns,
            exclude_patterns=self.exclude_patterns,
            max_memory_fraction=self.max_memory_fraction,
            force=self.force,
            progress=self._progress,
            chunk_size=self.chunk_size,
            workers=self.workers,
            max_memory_bytes=self.max_memory_bytes,
            reserve_bytes=self.reserve_bytes,
        )
        self.load_finished_at = time.time()
        build_view(self.cached, self.view, replace=self.replace)
        load_seconds = self.load_finished_at - self.load_started_at
        self._write_state(
            {
                "state": "ready",
                "load_started_at": self.load_started_at,
                "load_finished_at": self.load_finished_at,
                "load_seconds": load_seconds,
                "copied_bytes": self.cached.total_bytes,
                "copied_human": human_bytes(self.cached.total_bytes),
                "total_bytes": self.cached.total_bytes,
                "total_human": human_bytes(self.cached.total_bytes),
                "progress": 1.0,
                "file_count": len(self.cached.files),
                "workers": self.workers,
                "rate_bytes_per_second": self.cached.total_bytes / load_seconds if load_seconds > 0 else None,
            }
        )
        self.log(f"[ready] name={self.name} view={self.view} files={len(self.cached.files)} bytes={human_bytes(self.cached.total_bytes)}")

    def _status(self) -> dict[str, Any]:
        assert self.cached is not None
        return {
            "name": self.name,
            "identity": self.identity,
            "process_token": self.process_token,
            "pid": os.getpid(),
            "source": str(self.source),
            "upstream_source": str(self.upstream_source) if self.upstream_source else None,
            "local_cache_path": str(self.local_cache_path) if self.local_cache_path else None,
            "view": str(self.view),
            "socket": str(self.socket_path),
            "started_at": self.started_at,
            "uptime_seconds": time.time() - self.started_at,
            "load_seconds": (self.load_finished_at - self.load_started_at) if self.load_started_at and self.load_finished_at else None,
            "file_count": len(self.cached.files),
            "total_bytes": self.cached.total_bytes,
            "total_human": human_bytes(self.cached.total_bytes),
            "include_patterns": list(self.include_patterns),
            "exclude_patterns": list(self.exclude_patterns),
            "workers": self.workers,
            "memory_budget": self._reservation.info if self._reservation else None,
            "sealed": True,
            "physical_residency_guaranteed": False,
            "idle_ttl": self.idle_ttl,
            "last_used_at": self.last_used_at,
            "active_leases": len(self._active_leases()),
            "reuse_requests": self.reuse_requests,
        }

    def _handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._management_lock:
            return self._handle_locked(payload)

    def _active_leases(self) -> dict:
        self._leases = {key: lease for key, lease in self._leases.items()
                        if process_matches(lease["pid"], lease["process_token"])}
        return self._leases

    def _touch(self) -> None:
        self.last_used_at, self._last_use = time.time(), time.monotonic()

    def _handle_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = payload.get("command")
        if self._stop and command not in ("ping", "status", "lease-release"):
            raise RuntimeError("daemon is stopping; retry after it exits")
        if command == "ping":
            return {"ok": True, "pong": True}
        if command == "status":
            return {"ok": True, "status": self._status()}
        if command == "manifest":
            assert self.cached is not None
            return {"ok": True, "manifest": self.cached.to_manifest(self.view)}
        if command == "touch":
            self._touch()
            self.reuse_requests += 1
            return {"ok": True}
        if command in ("lease-acquire", "lease-update"):
            pid, token = int(payload["pid"]), payload.get("process_token")
            if not process_matches(pid, token):
                raise RuntimeError("lease owner is not a live process")
            if len(self._active_leases()) >= 4096:
                raise RuntimeError("too many active leases")
            key = payload.get("lease") if command == "lease-update" else uuid.uuid4().hex
            if command == "lease-update" and key not in self._leases:
                raise RuntimeError("unknown lease")
            self._leases[key] = dict(pid=pid, process_token=token)
            self._touch()
            self.reuse_requests += command == "lease-acquire"
            return {"ok": True, "lease": key, "view": str(self.view)}
        if command == "lease-release":
            self._leases.pop(payload.get("lease"), None)
            self._touch()
            return {"ok": True}
        if command in ("stop", "evict"):
            if self._active_leases() and not payload.get("force", False):
                raise RuntimeError("cache has active leases; wait or use stop --force")
            if command == "evict" and self.last_used_at != payload.get("last_used_at"):
                raise RuntimeError("cache was used after collection was planned")
            self._remove_view_on_exit = bool(payload.get("remove_view", False))
            self._stop = True
            return {"ok": True, "stopping": True}
        return {"ok": False, "error": f"unknown command: {command!r}"}

    def _serve_connection(self, conn: socket.socket) -> None:
        try:
            with conn:
                conn.settimeout(1.0)
                if hasattr(socket, "SO_PEERCRED"):
                    _, uid, _ = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if uid != os.getuid():
                        return
                data = bytearray()
                deadline = time.monotonic() + 1.0
                while b"\n" not in data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    conn.settimeout(remaining)
                    chunk = conn.recv(min(65536 - len(data), 4096))
                    if not chunk:
                        return
                    data.extend(chunk)
                    if len(data) >= 65536:
                        return
                try:
                    payload = json.loads(data.split(b"\n", 1)[0])
                    if not isinstance(payload, dict):
                        raise ValueError("request must be a JSON object")
                    response = self._handle(payload)
                except (ValueError, TypeError, RuntimeError) as exc:
                    response = {"ok": False, "error": str(exc)}
                conn.settimeout(1.0)
                conn.sendall(json.dumps(response).encode("utf-8") + b"\n")
        except OSError:
            pass  # A stalled or disconnected client must not terminate the daemon.
        finally:
            self._connections.release()

    def _cleanup_socket(self) -> None:
        if not self._owns_runtime:
            return
        for path, identity in self._claimed_runtime.items():
            try:
                st = path.lstat()
                if (st.st_dev, st.st_ino) == identity:
                    path.unlink()
            except OSError:
                pass

    def _claim_runtime_file(self, path: Path) -> None:
        st = path.lstat()
        self._claimed_runtime[path] = (st.st_dev, st.st_ino)

    def _check_runtime_paths(self) -> None:
        paths = (self.socket_path, self.pid_path, self.state_path)
        if len({path.absolute() for path in paths}) != len(paths):
            raise RuntimeError("socket, PID and state paths must be distinct")
        for path in paths:
            for source in (self.source, self.upstream_source):
                if source and path.resolve().is_relative_to(source.resolve()):
                    raise RuntimeError(f"runtime files must be outside the model directory: {path}")
            if path.is_symlink():
                raise RuntimeError(f"refusing to replace a runtime symlink: {path}")
        if self.socket_path.exists() and not self.socket_path.is_socket():
            raise RuntimeError(f"refusing to replace a non-socket file: {self.socket_path}")
        for path in (self.pid_path, self.state_path):
            if path.exists() and not stat.S_ISREG(path.lstat().st_mode):
                raise RuntimeError(f"runtime metadata must be a regular file: {path}")
        for path in paths:
            if path.exists() and path.lstat().st_uid != os.getuid():
                raise RuntimeError(f"runtime path belongs to another user: {path}")
        state = read_json(self.state_path)
        if self.state_path.exists():
            if (not state or state.get("runtime_format") != 1 or state.get("name") != self.name
                    or state.get("socket") != str(self.socket_path) or not isinstance(state.get("pid"), int)):
                raise RuntimeError(f"refusing to overwrite unverified runtime state: {self.state_path}")
            if process_matches(state["pid"], state.get("process_token")):
                raise RuntimeError("runtime metadata belongs to a live daemon")
        if self.pid_path.exists():
            try:
                pid = int(self.pid_path.read_text().strip())
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"refusing to overwrite an unverified PID file: {self.pid_path}") from exc
            if not state or pid != state.get("pid") or state.get("runtime_format") != 1:
                raise RuntimeError(f"PID file has no matching owned state: {self.pid_path}")
        if self.socket_path.exists():
            st = self.socket_path.lstat()
            if not state or state.get("runtime_socket") != {"device": st.st_dev, "inode": st.st_ino}:
                raise RuntimeError(f"socket has no matching owned state: {self.socket_path}")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.settimeout(.2)
                try:
                    probe.connect(str(self.socket_path))
                except ConnectionRefusedError:
                    pass  # Only a verified, non-listening endpoint may be reclaimed.
                except OSError as exc:
                    raise RuntimeError(f"cannot establish that runtime socket is stale: {self.socket_path}") from exc
                else:
                    raise RuntimeError(f"runtime socket still has a listener: {self.socket_path}")

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: object) -> None:
        self.log(f"[signal] received {signum}, shutting down")
        if self.cached is None:
            self._cleanup_socket()
            os._exit(0)
        self._stop = True

    def serve_forever(self) -> int:
        with ExitStack() as locks:
            locks.enter_context(FileLock(self.socket_path.with_suffix(".owner.lock"), timeout=0))
            locks.enter_context(FileLock(self.view.parent / f".{self.view.name}.owner.lock", timeout=0))
            for path in (self.pid_path, self.state_path):
                locks.enter_context(FileLock(path.with_name(f".{path.name}.owner.lock"), timeout=0))
            self._check_runtime_paths()
            if self.local_cache_path:
                locks.enter_context(FileLock(lease_path(self.local_cache_path), shared=True))
                if not verify_cache(self.local_cache_path, full=False)["valid"]:
                    raise RuntimeError("local cache changed or was removed before daemon startup")
            files = collect_source_files(self.source, self.include_patterns, self.exclude_patterns)
            if not files:
                raise RuntimeError("no files selected")
            total = sum(f.size for f in files)
            buffers = min(self.workers, len(files)) * min(self.chunk_size, max(f.size for f in files))
            self._reservation = locks.enter_context(ResidentReservation(
                self.socket_path.parent, self.name, total, buffers,
                max_resident_bytes=self.max_resident_bytes, fraction=self.max_memory_fraction,
                max_bytes=self.max_memory_bytes, reserve_bytes=self.reserve_bytes, force=self.force))
            self._owns_runtime = True
            try:
                return self._serve_owned()
            finally:
                self._owns_runtime = False

    def _serve_owned(self) -> int:
        if not self.identity:
            self.identity = source_identity(self.upstream_source or self.source, self.include_patterns, self.exclude_patterns)
        self._install_signal_handlers()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._write_state({"state": "starting"})
        atomic_text(self.pid_path, str(os.getpid()))
        self._claim_runtime_file(self.pid_path)

        server: socket.socket | None = None
        clients = ThreadPoolExecutor(max_workers=4, thread_name_prefix="iloader-ipc")

        try:
            self.load()
            self._reservation.ready()
            self._touch()
            if source_identity(self.upstream_source or self.source, self.include_patterns, self.exclude_patterns) != self.identity:
                raise RuntimeError("source model changed during startup; stop and retry")

            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.settimeout(1.0)
            server.bind(str(self.socket_path))
            self._claim_runtime_file(self.socket_path)
            st = self.socket_path.lstat()
            self._socket_identity = {"device": st.st_dev, "inode": st.st_ino}
            os.chmod(self.socket_path, 0o600)
            server.listen(16)
            self._write_state({"state": "ready", "progress": 1.0,
                               "copied_bytes": self.cached.total_bytes, **self._status()})
            self.log(f"[listen] socket={self.socket_path}")

            while not self._stop:
                with self._management_lock:
                    if self.idle_ttl is not None and not self._active_leases() and time.monotonic() - self._last_use >= self.idle_ttl:
                        self._stop, self._remove_view_on_exit = True, True
                        break
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                if self._connections.acquire(blocking=False):
                    clients.submit(self._serve_connection, conn)
                else:
                    conn.close()
        finally:
            clients.shutdown(wait=True)
            if server is not None:
                server.close()
            self._cleanup_socket()
            if self.cached is not None:
                self.cached.close()
            if self._remove_view_on_exit:
                try:
                    _safe_remove_view(self.view, expected_pid=os.getpid())
                except (OSError, RuntimeError) as exc:
                    self.log(f"[cleanup] {exc}")
            self.log("[stop] daemon exited")
        return 0
