from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import __version__
from .benchmark import benchmark_path
from .cache import DEFAULT_EXCLUDES, build_view, human_bytes, inspect_source, load_model_into_memory, source_identity, _safe_remove_view
from .runtime import FileLock, process_matches, process_token, read_json, signal_owned_process, validate_name
from .memory import parse_bytes
from .daemon import WeightLoaderDaemon
from .distributed import build_tp_plan, create_rank_views, simulate_distributed_read
from .ipc import LoaderConnectionError, request
from .local_cache import cache_status, list_model_caches, materialize_model_cache, verify_cache, garbage_collect, lease_path, touch_cache
from .paths import (
    default_log_path,
    default_pid_path,
    default_socket_path,
    default_state_path,
    default_view_path,
    runtime_dir,
)


def _patterns(values: list[str] | None) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    return tuple(values or ())


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _socket(args: argparse.Namespace) -> Path:
    return Path(args.socket).expanduser() if getattr(args, "socket", None) else default_socket_path(args.name, getattr(args, "base_dir", None))


def _pid(args: argparse.Namespace) -> Path:
    return Path(args.pid_file).expanduser() if getattr(args, "pid_file", None) else default_pid_path(args.name, getattr(args, "base_dir", None))


def _state(args: argparse.Namespace) -> Path:
    return Path(args.state_file).expanduser() if getattr(args, "state_file", None) else default_state_path(args.name, getattr(args, "base_dir", None))


def _view(args: argparse.Namespace) -> Path:
    return Path(args.view).expanduser() if getattr(args, "view", None) else default_view_path(args.name, getattr(args, "base_dir", None))


def _resolve_name(args: argparse.Namespace) -> str:
    explicit = getattr(args, "name", None)
    positional = getattr(args, "name_arg", None)
    if explicit and positional and explicit != positional:
        raise ValueError(f"conflicting names: {positional!r} and --name {explicit!r}")
    name = explicit or positional
    if not name:
        raise ValueError("missing model name; pass --name NAME")
    args.name = name
    return name


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except ValueError:
        return False


def _starting_status(args: argparse.Namespace, error: Exception) -> dict | None:
    pid_path = _pid(args)
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if not _pid_is_running(pid):
        return None

    log_path = Path(getattr(args, "log_file", None)).expanduser() if getattr(args, "log_file", None) else default_log_path(args.name, getattr(args, "base_dir", None))
    try:
        uptime = time.time() - pid_path.stat().st_mtime
    except OSError:
        uptime = None
    state = _load_state(_state(args))
    result = {
        "name": args.name,
        "pid": pid,
        "state": "starting",
        "socket": str(_socket(args)),
        "view": str(_view(args)),
        "pid_file": str(pid_path),
        "log": str(log_path),
        "uptime_seconds": uptime,
        "message": f"daemon process is running but the status socket is not ready yet: {error}",
    }
    if state:
        result.update(state)
        result["state"] = state.get("state") or "starting"
        result["pid"] = pid
    if not state or not process_matches(pid, state.get("process_token")):
        return None
    return result


def _load_state(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _progress_line(state: dict) -> str | None:
    total = int(state.get("total_bytes") or 0)
    copied = int(state.get("copied_bytes") or 0)
    if total <= 0:
        status = state.get("state") or "loading"
        return f"[load] {status}: waiting for file scan"
    progress = min(max(copied / total, 0.0), 1.0)
    width = 24
    filled = min(width, int(progress * width))
    bar = "#" * filled + "-" * (width - filled)
    rate = state.get("rate_human_per_second")
    if not rate and state.get("rate_bytes_per_second"):
        rate = f"{human_bytes(int(state['rate_bytes_per_second']))}/s"
    eta = _format_duration(0 if copied >= total else state.get("eta_seconds"))
    current = state.get("current_file")
    suffix = f" last={current}" if current else ""
    return (
        f"[load] [{bar}] {progress * 100:6.2f}% "
        f"{human_bytes(copied)} / {human_bytes(total)}"
        f" rate={rate or '?'} eta={eta}{suffix}"
    )


def _print_progress_update(args: argparse.Namespace, previous: str | None) -> str | None:
    state = _load_state(_state(args))
    if not state:
        return previous
    line = _progress_line(state)
    if not line or line == previous:
        return previous
    if sys.stderr.isatty():
        print(f"\r{line}", end="", file=sys.stderr, flush=True)
    else:
        print(line, file=sys.stderr)
    return line


def _cache_progress(copied: int, total: int, rel: Path, _reused: bool = False) -> None:
    pct = copied / total * 100 if total else 100.0
    print(f"[cache] {pct:6.2f}% {human_bytes(copied)} / {human_bytes(total)} copied; last={rel.as_posix()}", file=sys.stderr)


def _prepare_model_source(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    original = Path(args.model).expanduser().resolve()
    if not getattr(args, "local_cache", False):
        upstream = Path(args.upstream_source).expanduser().resolve() if getattr(args, "upstream_source", None) else None
        local_cache_path = Path(args.local_cache_path).expanduser().resolve() if getattr(args, "local_cache_path", None) else None
        return original, upstream, local_cache_path

    result = materialize_model_cache(
        original,
        name=getattr(args, "cache_name", None) or getattr(args, "name", None),
        cache_root=getattr(args, "cache_root", None),
        base_dir=getattr(args, "base_dir", None),
        include_patterns=_patterns(getattr(args, "pattern", None)),
        exclude_patterns=tuple(getattr(args, "exclude", None) or DEFAULT_EXCLUDES),
        refresh=getattr(args, "refresh_cache", False),
        workers=getattr(args, "cache_workers", 4),
        chunk_size=getattr(args, "chunk_size", 64 * 1024 * 1024),
        progress=None if getattr(args, "quiet", False) else _cache_progress,
    )
    action = "hit" if result.reused else "built"
    print(
        f"[cache] {action}: {result.cache_path} ({human_bytes(result.total_bytes)}, {result.file_count} files, {result.elapsed_seconds:.3f}s)",
        file=sys.stderr,
    )
    return result.cache_path, original, result.cache_path


def cmd_inspect(args: argparse.Namespace) -> int:
    data = inspect_source(Path(args.model), _patterns(args.pattern), tuple(args.exclude or DEFAULT_EXCLUDES))
    if args.json:
        _print_json(data)
    else:
        print(f"source: {data['source_root']}")
        print(f"files:  {data['file_count']}")
        print(f"bytes:  {data['total_human']}")
        available = data.get("available_memory_bytes")
        if available:
            print(f"mem:    {human_bytes(available)} available")
        print("suffixes:")
        for suffix, item in sorted(data["by_suffix"].items(), key=lambda kv: kv[1]["bytes"], reverse=True):
            print(f"  {suffix:16s} {item['count']:6d} {human_bytes(item['bytes'])}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    source, upstream_source, local_cache_path = _prepare_model_source(args)
    daemon = WeightLoaderDaemon(
        name=args.name,
        source=source,
        view=_view(args),
        socket_path=_socket(args),
        pid_path=_pid(args),
        state_path=_state(args),
        upstream_source=upstream_source,
        local_cache_path=local_cache_path,
        include_patterns=_patterns(args.pattern),
        exclude_patterns=tuple(args.exclude or DEFAULT_EXCLUDES),
        replace=args.replace,
        force=args.force,
        max_memory_fraction=args.max_memory_fraction,
        chunk_size=args.chunk_size,
        workers=args.workers,
        quiet=args.quiet,
        identity=json.loads(args.identity) if getattr(args, "identity", None) else None,
        max_memory_bytes=args.max_memory_bytes,
        reserve_bytes=args.reserve_bytes,
        max_resident_bytes=args.max_resident_bytes,
        idle_ttl=args.idle_ttl,
    )
    return daemon.serve_forever()


def _daemon_command(args: argparse.Namespace, model_path: Path | None = None, upstream_source: Path | None = None, local_cache_path: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "infiniloader",
        "_serve",
        str(model_path or Path(args.model).expanduser().resolve()),
        "--name",
        args.name,
        "--view",
        str(_view(args)),
        "--socket",
        str(_socket(args)),
        "--pid-file",
        str(_pid(args)),
        "--state-file",
        str(_state(args)),
        "--chunk-size",
        str(args.chunk_size),
        "--workers",
        str(args.workers),
        "--max-memory-fraction",
        str(args.max_memory_fraction),
        "--replace",
        "--quiet",
    ]
    if upstream_source:
        command.extend(["--upstream-source", str(upstream_source)])
    if local_cache_path:
        command.extend(["--local-cache-path", str(local_cache_path)])
    for pattern in args.pattern or []:
        command.extend(["--pattern", pattern])
    for pattern in args.exclude or []:
        command.extend(["--exclude", pattern])
    if args.force:
        command.append("--force")
    command.extend(["--identity", json.dumps(args.identity, sort_keys=True)])
    command.extend(["--reserve-bytes", str(args.reserve_bytes)])
    for option in ("max_memory_bytes", "max_resident_bytes"):
        if getattr(args, option, None) is not None:
            command.extend(["--" + option.replace("_", "-"), str(getattr(args, option))])
    if args.idle_ttl is not None:
        command.extend(["--idle-ttl", str(args.idle_ttl)])
    return command


def cmd_start(args: argparse.Namespace) -> int:
    args.identity = source_identity(Path(args.model), _patterns(args.pattern), tuple(args.exclude or DEFAULT_EXCLUDES))
    with FileLock(_socket(args).with_suffix(".start.lock"), timeout=args.wait_timeout):
        return _start_locked(args)


def _require_identity(args: argparse.Namespace, status: dict) -> None:
    if status.get("identity") != args.identity:
        raise RuntimeError("name already belongs to a different model, revision or file selection; use another name or stop it first")
    if args.view and Path(status["view"]).absolute() != Path(args.view).expanduser().absolute():
        raise RuntimeError("name already belongs to a different view path")
    if args.refresh_cache:
        raise RuntimeError("stop the running cache before requesting --refresh-cache")
    if args.idle_ttl is not None and status.get("idle_ttl") != args.idle_ttl:
        raise RuntimeError("stop the running cache before changing --idle-ttl")
    for option in ("max_memory_bytes", "max_resident_bytes"):
        requested = getattr(args, option, None)
        if requested is not None and (status.get("memory_budget") or {}).get(option) != requested:
            raise RuntimeError(f"stop the running cache before changing --{option.replace('_', '-')}")


def _start_locked(args: argparse.Namespace) -> int:
    sock = _socket(args)
    try:
        status = request(sock, {"command": "status"}, timeout=1.0)["status"]
    except LoaderConnectionError:
        status = None
    if status is not None:
        _require_identity(args, status)
        request(sock, {"command": "touch"})
        print(status["view"])
        print(f"already running: name={args.name} pid={status['pid']}", file=sys.stderr)
        return 0
    starting = _starting_status(args, LoaderConnectionError("not ready"))
    proc = None
    if starting:
        _require_identity(args, starting)
        if args.no_wait:
            print(starting["view"])
            return 0
    else:
        source, upstream_source, local_cache_path = _prepare_model_source(args)
        if source_identity(Path(args.model), _patterns(args.pattern), tuple(args.exclude or DEFAULT_EXCLUDES)) != args.identity:
            raise RuntimeError("source model changed while preparing local cache; retry")
        log_path = Path(args.log_file).expanduser() if args.log_file else default_log_path(args.name, getattr(args, "base_dir", None))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log:
            proc = subprocess.Popen(
                _daemon_command(args, source, upstream_source, local_cache_path),
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, close_fds=True,
            )
        print(f"starting name={args.name} pid={proc.pid} log={log_path}", file=sys.stderr)

    deadline = time.monotonic() + args.wait_timeout
    last_progress = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"daemon exited with code {proc.returncode}; see {log_path}")
        if args.no_wait:
            registered = _starting_status(args, LoaderConnectionError("not ready"))
            if registered and registered["pid"] == proc.pid:
                _require_identity(args, registered)
                print(registered["view"])
                return 0
        try:
            status = request(sock, {"command": "status"}, timeout=1.0)["status"]
        except LoaderConnectionError:
            last_progress = _print_progress_update(args, last_progress)
            if proc is None and not _starting_status(args, LoaderConnectionError("not ready")):
                raise RuntimeError("daemon exited before becoming ready")
            time.sleep(0.1)
            continue
        _require_identity(args, status)
        request(sock, {"command": "touch"})
        if sys.stderr.isatty() and last_progress:
            print(file=sys.stderr)
        print(status["view"])
        return 0
    raise TimeoutError("timeout waiting for daemon readiness; inspect status and log before retrying")


def cmd_status(args: argparse.Namespace) -> int:
    _resolve_name(args)
    try:
        status = request(_socket(args), {"command": "status"}, timeout=args.timeout)["status"]
    except LoaderConnectionError as exc:
        starting = _starting_status(args, exc)
        if starting:
            if args.json:
                _print_json(starting)
            else:
                print(f"name:   {starting['name']}")
                print(f"state:  {starting['state']}")
                print(f"pid:    {starting['pid']}")
                print(f"socket: {starting['socket']}")
                print(f"view:   {starting['view']}")
                print(f"log:    {starting['log']}")
                line = _progress_line(starting)
                if line:
                    print(f"progress: {line.removeprefix('[load] ')}")
                if starting.get("uptime_seconds") is not None:
                    print(f"uptime: {starting['uptime_seconds']:.1f}s")
                print("note:   daemon is still loading; the status socket will appear after the view is ready")
            return 0
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        _print_json(status)
    else:
        print(f"name:   {status['name']}")
        print(f"pid:    {status['pid']}")
        print(f"source: {status['source']}")
        if status.get("upstream_source"):
            print(f"origin: {status['upstream_source']}")
        if status.get("local_cache_path"):
            print(f"cache:  {status['local_cache_path']}")
        print(f"view:   {status['view']}")
        print(f"files:  {status['file_count']}")
        print(f"bytes:  {status['total_human']}")
        print(f"workers:{status.get('workers', 1)}")
        if status.get("load_seconds") is not None:
            print(f"load:   {status['load_seconds']:.3f}s")
        print(f"uptime: {status['uptime_seconds']:.1f}s")
        print(f"leases: {status.get('active_leases', 0)}; reuse requests: {status.get('reuse_requests', 0)}")
        print(f"sealed: {status.get('sealed', False)}; physical residency guaranteed: {status.get('physical_residency_guaranteed', False)}")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    _resolve_name(args)
    status = request(_socket(args), {"command": "status"}, timeout=args.timeout)["status"]
    print(status["view"])
    return 0


def _status_for_name(name: str, args: argparse.Namespace) -> dict:
    ns = argparse.Namespace(
        name=name,
        socket=None,
        pid_file=None,
        base_dir=getattr(args, "base_dir", None),
        view=None,
        log_file=None,
        timeout=getattr(args, "timeout", 1.0),
    )
    try:
        status = request(_socket(ns), {"command": "status"}, timeout=ns.timeout)["status"]
        status["state"] = "ready"
        return status
    except LoaderConnectionError as exc:
        starting = _starting_status(ns, exc)
        if starting:
            return starting
        try:
            pid = int(_pid(ns).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None
        return {
            "name": name,
            "state": "stale" if pid is not None else "down",
            "pid": pid,
            "socket": str(_socket(ns)),
            "view": str(_view(ns)),
            "log": str(default_log_path(name, getattr(args, "base_dir", None))),
            "message": str(exc),
        }


def _active_daemon_rows(args: argparse.Namespace) -> list[dict]:
    run = runtime_dir(getattr(args, "base_dir", None))
    rows = []
    for pid_path in sorted(run.glob("*.pid")):
        rows.append(_status_for_name(pid_path.name[:-4], args))
    return rows


def cmd_list(args: argparse.Namespace) -> int:
    rows = _active_daemon_rows(args)
    if args.names_only:
        for row in rows:
            if row.get("state") != "stale":
                print(row["name"])
        return 0
    if args.json:
        _print_json(rows)
        return 0
    if not rows:
        print("no active caches")
        return 0

    print(f"{'name':24s} {'state':10s} {'pid':>8s} {'bytes':>12s} view")
    for row in rows:
        pid = str(row.get("pid") or "-")
        total = row.get("total_human") or "-"
        print(f"{row['name'][:24]:24s} {row.get('state', '-')[:10]:10s} {pid:>8s} {total:>12s} {row.get('view', '-')}")
    return 0


def _stop_one(args: argparse.Namespace) -> int:
    _resolve_name(args)
    try:
        response = request(_socket(args), {"command": "stop", "remove_view": args.remove_view,
                                          "force": getattr(args, "force", False)}, timeout=args.timeout)
        print(f"{args.name}: {'stopping' if response.get('stopping') else 'ok'}")
        return 0
    except LoaderConnectionError as exc:
        starting = _starting_status(args, exc)
        if not starting:
            print(f"{args.name}: {exc}", file=sys.stderr)
            return 1
        if starting.get("state") not in ("starting", "loading") and not getattr(args, "force", False):
            raise RuntimeError("ready daemon is unresponsive; no signal was sent, retry or use stop --force")
        signal_owned_process(int(starting["pid"]), starting.get("process_token"))
        if args.remove_view:
            deadline = time.monotonic() + args.timeout
            while process_matches(int(starting["pid"]), starting.get("process_token")):
                if time.monotonic() >= deadline:
                    raise TimeoutError("daemon has not exited; view was preserved")
                time.sleep(.05)
            view = Path(starting["view"])
            if view.exists():
                _safe_remove_view(view, expected_pid=int(starting["pid"]))
        print(f"{args.name}: stopping pid={starting['pid']}")
        return 0


def cmd_stop(args: argparse.Namespace) -> int:
    if getattr(args, "all", False):
        rows = _active_daemon_rows(args)
        if not rows:
            print("no active caches")
            return 0
        rc = 0
        for row in rows:
            if row.get("state") == "stale":
                print(f"{row['name']}: stale pid file; skipping", file=sys.stderr)
                rc = max(rc, 1)
                continue
            ns = argparse.Namespace(
                name=row["name"],
                name_arg=None,
                socket=None,
                pid_file=None,
                base_dir=args.base_dir,
                view=None,
                timeout=args.timeout,
                remove_view=args.remove_view,
                force=getattr(args, "force", False),
            )
            try:
                rc = max(rc, _stop_one(ns))
            except (RuntimeError, OSError, ValueError) as exc:
                print(f"{row['name']}: {exc}", file=sys.stderr)
                rc = max(rc, 1)
        return rc
    return _stop_one(args)


def cmd_manifest(args: argparse.Namespace) -> int:
    _resolve_name(args)
    manifest = request(_socket(args), {"command": "manifest"}, timeout=args.timeout)["manifest"]
    _print_json(manifest)
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    result = materialize_model_cache(
        Path(args.model),
        name=args.cache_name or args.name,
        cache_root=args.cache_root,
        base_dir=args.base_dir,
        include_patterns=_patterns(args.pattern),
        exclude_patterns=tuple(args.exclude or DEFAULT_EXCLUDES),
        refresh=args.refresh,
        workers=args.cache_workers,
        chunk_size=args.chunk_size,
        progress=None if args.quiet else _cache_progress,
    )
    data = result.to_dict()
    if args.json:
        _print_json(data)
    else:
        print(data["cache_path"])
        print(f"source: {data['source_root']}", file=sys.stderr)
        print(f"files:  {data['file_count']}", file=sys.stderr)
        print(f"bytes:  {data['total_human']}", file=sys.stderr)
        print(f"reused: {data['reused']}", file=sys.stderr)
        print(f"time:   {data['elapsed_seconds']:.3f}s", file=sys.stderr)
    return 0


def cmd_cache_status(args: argparse.Namespace) -> int:
    data = cache_status(
        Path(args.model),
        name=args.cache_name or args.name,
        cache_root=args.cache_root,
        base_dir=args.base_dir,
        include_patterns=_patterns(args.pattern),
        exclude_patterns=tuple(args.exclude or DEFAULT_EXCLUDES),
    )
    if args.json:
        _print_json(data)
    else:
        print(f"source: {data['source_root']}")
        print(f"cache:  {data['cache_path']}")
        print(f"exists: {data['exists']}")
        print(f"valid:  {data['valid']}")
        print(f"files:  {data['file_count']}")
        print(f"bytes:  {data['total_human']}")
    return 0


def cmd_cache_list(args: argparse.Namespace) -> int:
    rows = list_model_caches(cache_root=args.cache_root, base_dir=args.base_dir)
    if args.json:
        _print_json(rows)
    else:
        if not rows:
            print("no caches")
            return 0
        for row in rows:
            print(f"{row['cache_path']}")
            print(f"  source: {row['source_root']}")
            print(f"  files:  {row['file_count']}")
            print(f"  bytes:  {row['total_human']}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().absolute()
    with FileLock(lease_path(path), shared=True):
        data = verify_cache(path, full=not args.quick)
    if args.json:
        _print_json(data)
    else:
        print(f"valid: {data['valid']} ({data['checked_files']} files)")
        for error in data["errors"]:
            print(error, file=sys.stderr)
    return 0 if data["valid"] else 1


def cmd_gc(args: argparse.Namespace) -> int:
    if args.max_bytes is not None and args.max_bytes < 0 or args.ttl is not None and args.ttl < 0:
        raise ValueError("max-bytes and ttl must be non-negative")
    if args.resident and args.cache_root:
        raise ValueError("--cache-root applies to disk caches; use --base-dir for resident caches")
    data = _gc_resident(args) if args.resident else garbage_collect(
        cache_root=args.cache_root, base_dir=args.base_dir,
        max_bytes=args.max_bytes, ttl=args.ttl, apply=args.apply)
    if args.json:
        _print_json(data)
    else:
        print("removed:" if args.apply else "would remove:")
        for row in data["candidates"]:
            print(f"  {row['cache_path']} ({row['total_human']})")
        for row in data["skipped"]:
            print(f"  skipped: {row['cache_path']} ({row['reason']})")
        print(f"remaining: {human_bytes(data['remaining_bytes'])}; budget met: {data['budget_met']}")
    return 0 if data["budget_met"] else 1


def _gc_resident(args: argparse.Namespace) -> dict:
    rows = _active_daemon_rows(args)
    total = sum(row.get("total_bytes", 0) for row in rows)
    candidates, skipped = [], []
    for row in sorted(rows, key=lambda r: r.get("last_used_at", float("inf"))):
        expired = args.ttl is not None and time.time() - row.get("last_used_at", time.time()) >= args.ttl
        if not expired and not (args.max_bytes is not None and total > args.max_bytes):
            continue
        if row.get("state") != "ready" or row.get("active_leases"):
            skipped.append(dict(cache_path=row.get("view", row["name"]), reason="loading, unavailable or leased"))
            continue
        if args.apply:
            try:
                request(row["socket"], {"command": "evict", "last_used_at": row["last_used_at"], "remove_view": True})
            except (LoaderConnectionError, RuntimeError) as exc:
                skipped.append(dict(cache_path=row["view"], reason=str(exc)))
                continue
        candidates.append(dict(cache_path=row["view"], total_human=row["total_human"]))
        total -= row["total_bytes"]
    return dict(apply=args.apply, candidates=candidates, skipped=skipped, remaining_bytes=total,
                budget_met=args.max_bytes is None or total <= args.max_bytes)


def cmd_doctor(args: argparse.Namespace) -> int:
    from .diagnostics import diagnose

    data = diagnose(Path(args.base_dir) if args.base_dir else None)
    if args.json:
        _print_json(data)
    else:
        for check in data["checks"]:
            print(f"{'ok' if check['ok'] else 'FAIL'}: {check['name']}: {check['detail']}")
        print(f"effective memory headroom: {data['memory']['available_bytes']} bytes")
        for warning in data["warnings"]:
            print(f"note: {warning}")
    return 0 if data["ok"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    program = args.program[1:] if args.program[:1] == ["--"] else args.program
    if args.cache_path and args.name_arg:
        program = [args.name_arg] + program
    if not program or not (args.name_arg or args.cache_path):
        raise ValueError("use run NAME -- COMMAND ... or run --cache-path PATH -- COMMAND ...")
    if args.cache_path:
        path = Path(args.cache_path).expanduser().resolve()
        with FileLock(lease_path(path), shared=True) as lease:
            if not verify_cache(path, full=False)["valid"]:
                raise RuntimeError("cache is missing or invalid; run verify or rebuild it")
            touch_cache(path)
            return _run_program(program, path, pass_fds=(lease.fd,))
    args.name = _resolve_name(args)
    sock = _socket(args)
    lease = request(sock, {"command": "lease-acquire", "pid": os.getpid(), "process_token": process_token(os.getpid())})
    try:
        return _run_program(program, Path(lease["view"]), socket_path=sock, lease=lease["lease"])
    finally:
        try:
            request(sock, {"command": "lease-release", "lease": lease["lease"]})
        except (LoaderConnectionError, RuntimeError):
            pass


def _run_program(program, path, *, pass_fds=(), socket_path=None, lease=None) -> int:
    env = dict(os.environ, ILOADER_MODEL_PATH=str(path))
    proc = subprocess.Popen([part.replace("{model}", str(path)) for part in program], env=env, pass_fds=pass_fds)
    try:
        if lease:
            token = process_token(proc.pid)
            if proc.poll() is None:
                try:
                    request(socket_path, {"command": "lease-update", "lease": lease, "pid": proc.pid,
                                          "process_token": token})
                except RuntimeError:
                    if proc.poll() is None:
                        raise
        return proc.wait()
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise


BASH_COMPLETION = r'''_iloader_complete() {
  local cur prev sub commands opts
  COMPREPLY=()
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"
  commands="inspect start list status view manifest stop bench selftest cache cache-status cache-list verify gc run doctor dist-views dist-simulate dist-plan completion"

  if [[ ${COMP_CWORD} -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "${commands}" -- "${cur}") )
    return 0
  fi

  case "${prev}" in
    --name)
      COMPREPLY=( $(compgen -W "$(iloader list --names-only 2>/dev/null)" -- "${cur}") )
      return 0
      ;;
    --device)
      COMPREPLY=( $(compgen -W "cpu cuda cuda:0" -- "${cur}") )
      return 0
      ;;
    --base-dir|--view|--cache-root|--log-file|--pid-file|--socket|--rank-view-root)
      compopt -o filenames 2>/dev/null || true
      COMPREPLY=( $(compgen -f -- "${cur}") )
      return 0
      ;;
  esac

  sub="${COMP_WORDS[1]}"
  case "${sub}" in
    start)
      opts="--name --view --socket --pid-file --base-dir --log-file --pattern --exclude --force --no-wait --wait-timeout --chunk-size --workers --max-memory-fraction --max-memory-bytes --max-resident-bytes --reserve-bytes --idle-ttl --local-cache --cache-root --cache-name --refresh-cache --cache-workers --help"
      ;;
    list)
      opts="--base-dir --timeout --json --names-only --help"
      ;;
    status|view|manifest)
      opts="--name --socket --base-dir --timeout --json --help"
      ;;
    stop)
      opts="--name --all --socket --base-dir --timeout --remove-view --force --help"
      ;;
    cache|cache-status)
      opts="--name --cache-name --cache-root --base-dir --pattern --exclude --refresh --cache-workers --chunk-size --quiet --json --help"
      ;;
    cache-list)
      opts="--cache-root --base-dir --json --help"
      ;;
    verify)
      opts="--quick --json --help"
      ;;
    gc)
      opts="--base-dir --cache-root --max-bytes --ttl --resident --apply --json --help"
      ;;
    run)
      opts="--base-dir --cache-path --help"
      ;;
    doctor)
      opts="--base-dir --json --help"
      ;;
    dist-views)
      opts="--name --ranks --base-dir --replace --json --help"
      ;;
    dist-simulate)
      opts="--ranks --rank-view-root --pattern --exclude --chunk-size --timeout --json --help"
      ;;
    dist-plan)
      opts="--tp --limit --json --help"
      ;;
    completion)
      opts="bash --help"
      ;;
    *)
      opts="--help --json --base-dir --name"
      ;;
  esac

  if [[ "${cur}" == -* ]]; then
    COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
    return 0
  fi
}
complete -F _iloader_complete iloader
'''


def cmd_completion(args: argparse.Namespace) -> int:
    if args.shell == "bash":
        print(BASH_COMPLETION)
        return 0
    raise ValueError(f"unsupported shell: {args.shell}")


def cmd_dist_views(args: argparse.Namespace) -> int:
    manifest = create_rank_views(
        Path(args.model),
        name=args.name,
        ranks=args.ranks,
        base_dir=args.base_dir,
        replace=args.replace,
    )
    if args.json:
        _print_json(manifest)
    else:
        print(Path(manifest["rank_paths"][0]).parent)
        print(f"source: {manifest['source']}", file=sys.stderr)
        print(f"ranks:  {manifest['ranks']}", file=sys.stderr)
        print(f"files:  {manifest['file_count']}", file=sys.stderr)
        print(f"bytes:  {manifest['total_human']}", file=sys.stderr)
        for path in manifest["rank_paths"]:
            print(f"rank:   {path}", file=sys.stderr)
    return 0


def cmd_dist_simulate(args: argparse.Namespace) -> int:
    result = simulate_distributed_read(
        Path(args.model),
        ranks=args.ranks,
        rank_view_root=Path(args.rank_view_root).expanduser() if args.rank_view_root else None,
        include_patterns=_patterns(args.pattern) or ("*.safetensors",),
        exclude_patterns=tuple(args.exclude or DEFAULT_EXCLUDES),
        chunk_size=args.chunk_size,
        timeout=args.timeout,
    )
    if args.json:
        _print_json(result)
    else:
        print(f"ok:        {result['ok']}")
        print(f"ranks:     {result['ranks']}")
        print(f"wall:      {result['wall_seconds']:.3f}s")
        print(f"aggregate: {result['aggregate_human']} @ {result['aggregate_gib_per_second']:.3f} GiB/s")
        for row in result["rank_results"]:
            if row.get("ok"):
                print(f"rank {row['rank']}: {row['seconds']:.3f}s {human_bytes(row['bytes'])} path={row['path']}")
            else:
                print(f"rank {row['rank']}: failed {row.get('error')} path={row['path']}")
    return 0 if result["ok"] else 1


def cmd_dist_plan(args: argparse.Namespace) -> int:
    plan = build_tp_plan(Path(args.model), args.tp)
    if args.json:
        _print_json(plan)
    else:
        print(f"model:       {plan['model_dir']}")
        print(f"tp:          {plan['tp']}")
        print(f"tensors:     {plan['tensor_count']}")
        print(f"bytes:       {plan['total_human']}")
        print(f"split:       {plan['split_tensor_count']} tensors, {plan['split_human']}")
        print(f"nondiv:      {plan['nondivisible_count']}")
        print("no-adapter rank-sharded HF emit:", plan["can_emit_rank_sharded_hf_dir_without_adapter"])
        print(plan["note"])
        shown = 0
        for item in plan["tensors"]:
            if item["action"] != "split":
                continue
            print(
                f"- {item['name']}: dim={item['dim']} shape={item['shape']} "
                f"per_rank={item.get('per_rank_shape')} divisible={item['divisible']}"
            )
            shown += 1
            if shown >= args.limit:
                remaining = plan["split_tensor_count"] - shown
                if remaining > 0:
                    print(f"... {remaining} more split candidates")
                break
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    result = benchmark_path(
        Path(args.path),
        include_patterns=_patterns(args.pattern),
        exclude_patterns=tuple(args.exclude or DEFAULT_EXCLUDES),
        repeat=args.repeat,
        chunk_size=args.chunk_size,
    )
    if args.json:
        _print_json(result)
    else:
        print(f"path:  {result['path']}")
        print(f"files: {result['file_count']}")
        print(f"bytes: {result['total_human']}")
        for run in result["runs"]:
            print(f"run {run['run']}: {run['seconds']:.3f}s, {run['gib_per_second']:.3f} GiB/s")
        print(f"best:  {result['best_gib_per_second']:.3f} GiB/s")
        print(f"median: {result['median_seconds']:.3f}s; cache state: {result['cache_state']}")
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="iloader-selftest-") as tmp:
        root = Path(tmp) / "model"
        root.mkdir()
        (root / "config.json").write_text('{"model_type":"dummy"}\n', encoding="utf-8")
        (root / "tokenizer.json").write_text('{"dummy":true}\n', encoding="utf-8")
        data = os.urandom(8 * 1024 * 1024 + 123)
        (root / "model.safetensors").write_bytes(data)
        view = Path(tmp) / "view"
        local = materialize_model_cache(root, name="selftest", base_dir=tmp, workers=2, progress=None)
        assert (local.cache_path / "model.safetensors").read_bytes() == data
        local2 = materialize_model_cache(root, name="selftest", base_dir=tmp, workers=2, progress=None)
        assert local2.reused
        cached = load_model_into_memory(local.cache_path, workers=2)
        try:
            build_view(cached, view, replace=True)
            assert (view / "config.json").read_text(encoding="utf-8").startswith("{")
            assert (view / "model.safetensors").read_bytes() == data
            script = (
                "from pathlib import Path; import sys, mmap; "
                "p=Path(sys.argv[1])/'model.safetensors'; "
                "f=p.open('rb'); mm=mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ); "
                "print(len(mm), mm[:4].hex()); mm.close(); f.close()"
            )
            output = subprocess.check_output([sys.executable, "-c", script, str(view)], text=True).strip()
            print(f"selftest ok: {output}")
        finally:
            cached.close()
            shutil.rmtree(view, ignore_errors=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    prog = Path(sys.argv[0]).name
    if prog in {"__main__.py", "python", "python3"}:
        prog = "iloader"
    parser = argparse.ArgumentParser(prog=prog, description="InfiniLoader: LLM weight loading and residency manager")
    parser.add_argument("--version", action="version", version=f"InfiniLoader {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    inspect_p = sub.add_parser("inspect", help="inspect selected files and memory budget")
    inspect_p.add_argument("model")
    inspect_p.add_argument("--pattern", action="append", help="include glob; can be repeated")
    inspect_p.add_argument("--exclude", action="append", help="exclude glob; replaces defaults when provided")
    inspect_p.add_argument("--json", action="store_true")
    inspect_p.set_defaults(func=cmd_inspect)
    from .shards import configure_parser as configure_shards
    configure_shards(sub)

    serve_p = sub.add_parser("_serve", help=argparse.SUPPRESS)
    serve_p.add_argument("model")
    serve_p.add_argument("--name", required=True)
    serve_p.add_argument("--view")
    serve_p.add_argument("--socket")
    serve_p.add_argument("--pid-file")
    serve_p.add_argument("--state-file", help=argparse.SUPPRESS)
    serve_p.add_argument("--base-dir")
    serve_p.add_argument("--upstream-source", help=argparse.SUPPRESS)
    serve_p.add_argument("--local-cache-path", help=argparse.SUPPRESS)
    serve_p.add_argument("--identity", help=argparse.SUPPRESS)
    serve_p.add_argument("--pattern", action="append")
    serve_p.add_argument("--exclude", action="append")
    serve_p.add_argument("--replace", action="store_true")
    serve_p.add_argument("--force", action="store_true")
    serve_p.add_argument("--quiet", action="store_true")
    serve_p.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    serve_p.add_argument("--workers", type=int, default=1, help="parallel memfd preload workers")
    serve_p.add_argument("--max-memory-fraction", type=float, default=0.90)
    serve_p.add_argument("--local-cache", action="store_true", help="copy model to local cache before memfd preload")
    serve_p.add_argument("--cache-root")
    serve_p.add_argument("--cache-name")
    serve_p.add_argument("--refresh-cache", action="store_true")
    serve_p.add_argument("--cache-workers", type=int, default=4)
    serve_p.set_defaults(func=cmd_serve)
    sub._choices_actions = [action for action in sub._choices_actions if action.dest != "_serve"]

    start_p = sub.add_parser("start", help="start background daemon")
    start_p.add_argument("model")
    start_p.add_argument("--name", required=True)
    start_p.add_argument("--view")
    start_p.add_argument("--socket")
    start_p.add_argument("--pid-file")
    start_p.add_argument("--base-dir")
    start_p.add_argument("--log-file")
    start_p.add_argument("--pattern", action="append")
    start_p.add_argument("--exclude", action="append")
    start_p.add_argument("--force", action="store_true")
    start_p.add_argument("--no-wait", action="store_true")
    start_p.add_argument("--wait-timeout", type=float, default=600.0)
    start_p.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    start_p.add_argument("--workers", type=int, default=1, help="parallel memfd preload workers")
    start_p.add_argument("--max-memory-fraction", type=float, default=0.90)
    start_p.add_argument("--local-cache", action="store_true", help="copy model to local cache before memfd preload")
    start_p.add_argument("--cache-root")
    start_p.add_argument("--cache-name")
    start_p.add_argument("--refresh-cache", action="store_true")
    start_p.add_argument("--cache-workers", type=int, default=4)
    start_p.set_defaults(func=cmd_start)

    for memory_parser in (serve_p, start_p):
        memory_parser.add_argument("--max-memory-bytes", type=parse_bytes, help="per-cache preload cap including worker buffers")
        memory_parser.add_argument("--reserve-bytes", type=parse_bytes, default=256 * 1024 * 1024, help="headroom left for the engine; default 256 MiB")
        memory_parser.add_argument("--max-resident-bytes", type=parse_bytes, help="set persistent total resident quota for this base directory")
        memory_parser.add_argument("--idle-ttl", type=float, help="evict after this many idle seconds; use run to protect readers")

    cache_p = sub.add_parser("cache", help="materialize a local model cache")
    cache_p.add_argument("model")
    cache_p.add_argument("--name")
    cache_p.add_argument("--cache-name")
    cache_p.add_argument("--cache-root")
    cache_p.add_argument("--base-dir")
    cache_p.add_argument("--pattern", action="append")
    cache_p.add_argument("--exclude", action="append")
    cache_p.add_argument("--refresh", action="store_true")
    cache_p.add_argument("--cache-workers", type=int, default=4)
    cache_p.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    cache_p.add_argument("--quiet", action="store_true")
    cache_p.add_argument("--json", action="store_true")
    cache_p.set_defaults(func=cmd_cache)

    cache_status_p = sub.add_parser("cache-status", help="check whether a local model cache is valid")
    cache_status_p.add_argument("model")
    cache_status_p.add_argument("--name")
    cache_status_p.add_argument("--cache-name")
    cache_status_p.add_argument("--cache-root")
    cache_status_p.add_argument("--base-dir")
    cache_status_p.add_argument("--pattern", action="append")
    cache_status_p.add_argument("--exclude", action="append")
    cache_status_p.add_argument("--json", action="store_true")
    cache_status_p.set_defaults(func=cmd_cache_status)

    cache_list_p = sub.add_parser("cache-list", help="list local model caches")
    cache_list_p.add_argument("--cache-root")
    cache_list_p.add_argument("--base-dir")
    cache_list_p.add_argument("--json", action="store_true")
    cache_list_p.set_defaults(func=cmd_cache_list)

    verify_p = sub.add_parser("verify", help="verify a versioned local cache using SHA-256")
    verify_p.add_argument("path")
    verify_p.add_argument("--quick", action="store_true", help="check file metadata without reading all bytes")
    verify_p.add_argument("--json", action="store_true")
    verify_p.set_defaults(func=cmd_verify)

    gc_p = sub.add_parser("gc", help="preview or apply LRU/TTL collection of unleased disk caches")
    gc_p.add_argument("--base-dir")
    gc_p.add_argument("--cache-root")
    gc_p.add_argument("--max-bytes", type=parse_bytes, help="target total cache bytes; accepts GiB/MiB")
    gc_p.add_argument("--resident", action="store_true", help="collect resident daemons instead of disk caches")
    gc_p.add_argument("--ttl", type=float, help="collect caches unused for this many seconds")
    gc_p.add_argument("--apply", action="store_true", help="delete the selected caches; default is dry run")
    gc_p.add_argument("--json", action="store_true")
    gc_p.set_defaults(func=cmd_gc)

    run_p = sub.add_parser("run", help="run a foreground command while protecting its cache from eviction")
    run_p.add_argument("--base-dir")
    run_p.add_argument("--cache-path")
    run_p.add_argument("name_arg", nargs="?")
    run_p.add_argument("program", nargs=argparse.REMAINDER)
    run_p.set_defaults(func=cmd_run)

    doctor_p = sub.add_parser("doctor", help="check Linux, memfd, proc access, locks and resource limits")
    doctor_p.add_argument("--base-dir")
    doctor_p.add_argument("--json", action="store_true")
    doctor_p.set_defaults(func=cmd_doctor)

    list_p = sub.add_parser("list", help="list active resident caches")
    list_p.add_argument("--base-dir")
    list_p.add_argument("--timeout", type=float, default=1.0)
    list_p.add_argument("--json", action="store_true")
    list_p.add_argument("--names-only", action="store_true", help=argparse.SUPPRESS)
    list_p.set_defaults(func=cmd_list)

    dist_views_p = sub.add_parser("dist-views", help="create full HF-compatible model views for simulated ranks")
    dist_views_p.add_argument("model")
    dist_views_p.add_argument("--name", required=True)
    dist_views_p.add_argument("--ranks", type=int, required=True)
    dist_views_p.add_argument("--base-dir")
    dist_views_p.add_argument("--replace", action="store_true")
    dist_views_p.add_argument("--json", action="store_true")
    dist_views_p.set_defaults(func=cmd_dist_views)

    dist_sim_p = sub.add_parser("dist-simulate", help="simulate concurrent multi-rank model reads on this host")
    dist_sim_p.add_argument("model")
    dist_sim_p.add_argument("--ranks", type=int, required=True)
    dist_sim_p.add_argument("--rank-view-root", help="root containing rank0/rank1/... directories")
    dist_sim_p.add_argument("--pattern", action="append")
    dist_sim_p.add_argument("--exclude", action="append")
    dist_sim_p.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    dist_sim_p.add_argument("--timeout", type=float, default=60.0)
    dist_sim_p.add_argument("--json", action="store_true")
    dist_sim_p.set_defaults(func=cmd_dist_simulate)

    dist_plan_p = sub.add_parser("dist-plan", help="analyze safetensors metadata for TP split candidates")
    dist_plan_p.add_argument("model")
    dist_plan_p.add_argument("--tp", type=int, required=True)
    dist_plan_p.add_argument("--limit", type=int, default=40)
    dist_plan_p.add_argument("--json", action="store_true")
    dist_plan_p.set_defaults(func=cmd_dist_plan)

    for name, help_text, func in [
        ("status", "show daemon status", cmd_status),
        ("view", "print daemon view path", cmd_view),
        ("manifest", "print daemon manifest", cmd_manifest),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("name_arg", nargs="?")
        p.add_argument("--name")
        p.add_argument("--socket")
        p.add_argument("--base-dir")
        p.add_argument("--timeout", type=float, default=5.0)
        if name in {"status"}:
            p.add_argument("--json", action="store_true")
        p.set_defaults(func=func)

    stop_p = sub.add_parser("stop", help="stop daemon")
    stop_p.add_argument("name_arg", nargs="?")
    stop_p.add_argument("--name")
    stop_p.add_argument("--all", action="store_true", help="stop every active resident cache in the run directory")
    stop_p.add_argument("--socket")
    stop_p.add_argument("--base-dir")
    stop_p.add_argument("--timeout", type=float, default=5.0)
    stop_p.add_argument("--remove-view", action="store_true")
    stop_p.add_argument("--force", action="store_true", help="stop even when commands hold active leases")
    stop_p.set_defaults(func=cmd_stop)

    bench_p = sub.add_parser("bench", help="sequential read benchmark")
    bench_p.add_argument("path")
    bench_p.add_argument("--pattern", action="append")
    bench_p.add_argument("--exclude", action="append")
    bench_p.add_argument("--repeat", type=int, default=3)
    bench_p.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    bench_p.add_argument("--json", action="store_true")
    bench_p.set_defaults(func=cmd_bench)

    selftest_p = sub.add_parser("selftest", help="verify memfd view with open/stat/mmap")
    selftest_p.set_defaults(func=cmd_selftest)

    completion_p = sub.add_parser("completion", help="print shell completion script")
    completion_p.add_argument("shell", choices=["bash"])
    completion_p.set_defaults(func=cmd_completion)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        for attribute in ("workers", "cache_workers", "chunk_size", "ranks", "tp", "repeat", "timeout", "wait_timeout"):
            value = getattr(args, attribute, None)
            if value is not None and value <= 0:
                raise ValueError(f"{attribute} must be positive")
        for attribute in ("name", "name_arg"):
            if args.command == "run" and getattr(args, "cache_path", None) and attribute == "name_arg":
                continue
            if getattr(args, attribute, None):
                validate_name(getattr(args, attribute))
        for attribute in ("idle_ttl", "max_memory_bytes", "max_resident_bytes"):
            value = getattr(args, attribute, None)
            if value is not None and value <= 0:
                raise ValueError(f"{attribute} must be positive")
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
