from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from typing import Callable
import uuid
import math

from .cache import DEFAULT_EXCLUDES, SourceFile, collect_source_files, human_bytes
from .paths import model_cache_root
from .runtime import FileLock, atomic_json, read_json


@dataclass
class LocalCacheResult:
    source_root: Path
    cache_path: Path
    total_bytes: int
    file_count: int
    reused: bool
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return dict(source_root=str(self.source_root), cache_path=str(self.cache_path),
                    total_bytes=self.total_bytes, total_human=human_bytes(self.total_bytes),
                    file_count=self.file_count, reused=self.reused, elapsed_seconds=self.elapsed_seconds)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("._-")[:64] or "model"


def cache_key(source_root: Path, name: str | None = None) -> str:
    source_root = source_root.expanduser().resolve()
    return f"{_safe_name(name or source_root.name)}-{hashlib.sha256(str(source_root).encode()).hexdigest()[:12]}"


def _root(cache_root=None, base_dir=None) -> Path:
    root = Path(cache_root).expanduser().resolve() if cache_root else model_cache_root(base_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _manifest_path(path: Path) -> Path:
    return path / ".iloader-cache" / "manifest.json"


def _files_manifest(files: list[SourceFile]) -> list[dict]:
    return [dict(relative_path=f.rel_posix, size=f.size, mtime_ns=f.mtime_ns,
                 ctime_ns=f.ctime_ns, device=f.device, inode=f.inode) for f in files]


def _revision(files, include_patterns, exclude_patterns) -> str:
    data = dict(files=_files_manifest(files), include=list(include_patterns), exclude=list(exclude_patterns))
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]


def lease_path(cache_path: Path) -> Path:
    return cache_path.parent / ".leases" / f"{cache_path.name}.lock"


def touch_cache(cache_path: Path) -> None:
    atomic_json(cache_path.parent / ".usage" / f"{cache_path.name}.json", {"last_used_at": time.time()})


def _safe_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError("invalid relative path in cache manifest")
    path = root / rel
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("cache file escapes its generation directory")
    return path


def _hash_stream(stream) -> str:
    digest = hashlib.sha256()
    for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(data)
    return digest.hexdigest()


def verify_cache(cache_path: Path, *, full: bool = True) -> dict:
    cache_path = cache_path.expanduser().absolute()
    manifest = read_json(_manifest_path(cache_path))
    errors, checked = [], 0
    if cache_path.is_symlink() or not manifest or manifest.get("format_version") != 2:
        errors.append("not a version-2 InfiniLoader disk cache; rebuild legacy caches to verify")
    elif (not isinstance(manifest.get("source_root"), str)
          or type(manifest.get("file_count")) is not int or manifest["file_count"] < 1
          or type(manifest.get("total_bytes")) is not int or manifest["total_bytes"] < 0):
        errors.append("invalid cache manifest metadata")
    else:
        records = manifest.get("cached_files")
        if (not isinstance(records, list) or not records or len(records) != manifest.get("file_count")
                or not all(isinstance(item, dict) for item in records)):
            errors.append("incomplete file manifest")
        else:
            names = [item.get("relative_path") for item in records]
            if (any(type(item.get("size")) is not int or item["size"] < 0 for item in records)
                    or sum(item["size"] for item in records) != manifest["total_bytes"]):
                errors.append("cached file sizes do not match total_bytes")
            source_records = manifest.get("files")
            source_names = [item.get("relative_path") for item in source_records if isinstance(item, dict)] if isinstance(source_records, list) else []
            if (not all(isinstance(name, str) for name in names) or len(set(names)) != len(names)
                    or not isinstance(source_records, list)
                    or not all(isinstance(item, dict) for item in source_records)
                    or not all(isinstance(name, str) for name in source_names)
                    or sorted(names) != sorted(source_names)):
                errors.append("file manifest names are missing, duplicated or inconsistent")
                return dict(cache_path=str(cache_path), valid=False, full=full, checked_files=0, errors=errors)
            try:
                actual = {path.relative_to(cache_path).as_posix() for path in cache_path.rglob("*")
                          if path.is_symlink() or not path.is_dir()}
                expected = set(names) | {".iloader-cache/manifest.json"}
                if actual != expected:
                    errors.append(f"generation file set changed: extra={sorted(actual - expected)}, missing={sorted(expected - actual)}")
            except OSError as exc:
                errors.append(f"cannot inventory generation: {exc}")
            for item in records:
                try:
                    file = _safe_file(cache_path, item["relative_path"])
                    st = file.stat()
                    if not stat.S_ISREG(st.st_mode) or (st.st_size, st.st_mtime_ns, st.st_ctime_ns) != (
                            item["size"], item["mtime_ns"], item["ctime_ns"]):
                        raise ValueError("file metadata changed")
                    if full:
                        with file.open("rb") as stream:
                            digest = _hash_stream(stream)
                        if digest != item["sha256"]:
                            raise ValueError("SHA-256 mismatch")
                    checked += 1
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{item.get('relative_path', '?')}: {exc}")
    return dict(cache_path=str(cache_path), valid=not errors, full=full, checked_files=checked, errors=errors)


def _created_at(path: Path) -> float:
    value = (read_json(_manifest_path(path)) or {}).get("created_at", 0)
    try:
        parsed = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
        return parsed if math.isfinite(parsed) else 0.0
    except (ValueError, OverflowError):
        return 0.0


def cache_is_valid(source_root, cache_path, files, include_patterns, exclude_patterns) -> bool:
    manifest = read_json(_manifest_path(cache_path))
    if not manifest or manifest.get("source_root") != str(source_root):
        return False
    if manifest.get("include_patterns") != list(include_patterns) or manifest.get("exclude_patterns") != list(exclude_patterns):
        return False
    return manifest.get("files") == _files_manifest(files) and verify_cache(cache_path, full=False)["valid"]


def cache_path_for(source_root: Path, *, name=None, cache_root=None, base_dir=None,
                   include_patterns=(), exclude_patterns=DEFAULT_EXCLUDES) -> Path:
    source_root = source_root.expanduser().resolve()
    files = collect_source_files(source_root, include_patterns, exclude_patterns)
    root = _root(cache_root, base_dir)
    prefix = f"{cache_key(source_root, name)}-{_revision(files, include_patterns, exclude_patterns)}"
    for candidate in sorted(root.glob(f"{prefix}*"),
                            key=_created_at, reverse=True):
        if cache_is_valid(source_root, candidate, files, include_patterns, exclude_patterns):
            return candidate
    return root / prefix


def _copy_one(item: SourceFile, cache_path: Path, chunk_size: int) -> dict:
    dst = cache_path / item.rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest, copied = hashlib.sha256(), 0
    with item.path.open("rb", buffering=0) as src, dst.open("xb", buffering=0) as target:
        before = os.fstat(src.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (item.size, item.mtime_ns, item.ctime_ns, item.inode):
            raise RuntimeError(f"source changed before cache copy: {item.path}")
        while copied < item.size:
            data = src.read(min(chunk_size, item.size - copied))
            if not data:
                raise IOError(f"short copy: {item.path}")
            digest.update(data)
            remaining = memoryview(data)
            while remaining:
                written = target.write(remaining)
                if not written:
                    raise IOError(f"short write: {dst}")
                remaining = remaining[written:]
            copied += len(data)
            del remaining, data
        after = os.fstat(src.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError(f"source changed during cache copy: {item.path}")
        os.fsync(target.fileno())
    os.utime(dst, ns=(item.mtime_ns, item.mtime_ns))
    dst.chmod(0o444)
    st = dst.stat()
    return dict(relative_path=item.rel_posix, size=st.st_size, mtime_ns=st.st_mtime_ns,
                ctime_ns=st.st_ctime_ns, sha256=digest.hexdigest())


def materialize_model_cache(source_root: Path, *, name=None, cache_root=None, base_dir=None,
                            include_patterns=(), exclude_patterns=DEFAULT_EXCLUDES, refresh=False,
                            workers=4, chunk_size=64 * 1024 * 1024,
                            progress: Callable | None = None) -> LocalCacheResult:
    if workers < 1 or chunk_size < 1:
        raise ValueError("workers and chunk_size must be positive")
    start = time.perf_counter()
    source_root = source_root.expanduser().resolve()
    root = _root(cache_root, base_dir)
    with FileLock(root / ".cache.lock", timeout=600):
        files = collect_source_files(source_root, include_patterns, exclude_patterns)
        if not files:
            raise RuntimeError(f"no regular files selected under {source_root}")
        total = sum(item.size for item in files)
        path = cache_path_for(source_root, name=name, cache_root=root,
                              include_patterns=include_patterns, exclude_patterns=exclude_patterns)
        if not refresh and cache_is_valid(source_root, path, files, include_patterns, exclude_patterns):
            touch_cache(path)
            return LocalCacheResult(source_root, path, total, len(files), True, time.perf_counter() - start)
        if shutil.disk_usage(root).free < total:
            raise OSError("insufficient disk space for a new cache generation; inspect iloader gc")
        if path.exists():
            prefix = f"{cache_key(source_root, name)}-{_revision(files, include_patterns, exclude_patterns)}"
            path = root / f"{prefix}-r{uuid.uuid4().hex[:12]}"
        temporary = Path(tempfile.mkdtemp(prefix=".build-", dir=root))
        try:
            copied, records = 0, []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for item, record in zip(files, pool.map(lambda f: _copy_one(f, temporary, chunk_size), files)):
                    records.append(record)
                    copied += item.size
                    if progress:
                        progress(copied, total, item.rel, False)
            if collect_source_files(source_root, include_patterns, exclude_patterns) != files:
                raise RuntimeError("source model changed during cache copy")
            manifest = dict(format_version=2, created_at=time.time(), source_root=str(source_root),
                            file_count=len(files), total_bytes=total,
                            include_patterns=list(include_patterns), exclude_patterns=list(exclude_patterns),
                            files=_files_manifest(files), cached_files=records)
            atomic_json(_manifest_path(temporary), manifest)
            os.rename(temporary, path)
            touch_cache(path)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return LocalCacheResult(source_root, path, total, len(files), False, time.perf_counter() - start)


def cache_status(source_root: Path, *, name=None, cache_root=None, base_dir=None,
                 include_patterns=(), exclude_patterns=DEFAULT_EXCLUDES) -> dict:
    source_root = source_root.expanduser().resolve()
    files = collect_source_files(source_root, include_patterns, exclude_patterns)
    path = cache_path_for(source_root, name=name, cache_root=cache_root, base_dir=base_dir,
                          include_patterns=include_patterns, exclude_patterns=exclude_patterns)
    total = sum(f.size for f in files)
    return dict(source_root=str(source_root), cache_path=str(path), exists=path.exists(),
                valid=cache_is_valid(source_root, path, files, include_patterns, exclude_patterns),
                file_count=len(files), total_bytes=total, total_human=human_bytes(total))


def list_model_caches(*, cache_root=None, base_dir=None) -> list[dict]:
    root = _root(cache_root, base_dir)
    rows = []
    for path in sorted(root.iterdir()):
        if path.name.startswith(".") or path.is_symlink() or not path.is_dir():
            continue
        manifest = read_json(_manifest_path(path))
        if not manifest or manifest.get("format_version") not in (1, 2):
            continue
        usage = read_json(root / ".usage" / f"{path.name}.json") or {}
        # Collection accounts actual model files, even when a damaged manifest lies
        # about total_bytes or extra files have appeared after publication.
        try:
            total = sum(file.lstat().st_size for file in path.rglob("*")
                        if file.is_file() and not file.is_symlink() and file != _manifest_path(path))
        except OSError:
            continue
        rows.append(dict(cache_path=str(path), source_root=manifest.get("source_root"),
                         file_count=manifest.get("file_count"), total_bytes=total, total_human=human_bytes(total),
                         format_version=manifest["format_version"], created_at=_created_at(path),
                         last_used_at=usage.get("last_used_at", _created_at(path))))
    return rows


def garbage_collect(*, cache_root=None, base_dir=None, max_bytes=None, ttl=None, apply=False) -> dict:
    if max_bytes is not None and max_bytes < 0 or ttl is not None and ttl < 0:
        raise ValueError("max_bytes and ttl must be non-negative")
    root = _root(cache_root, base_dir)
    removed, skipped = [], []
    with FileLock(root / ".cache.lock", timeout=30):
        rows = sorted(list_model_caches(cache_root=root), key=lambda row: row["last_used_at"])
        total, now = sum(row["total_bytes"] for row in rows), time.time()
        for row in rows:
            expired = ttl is not None and now - row["last_used_at"] >= ttl
            over_budget = max_bytes is not None and total > max_bytes
            if not (expired or over_budget):
                continue
            path = Path(row["cache_path"])
            try:
                with FileLock(lease_path(path), timeout=0):
                    if path.is_symlink() or path.parent != root or not read_json(_manifest_path(path)):
                        raise RuntimeError("cache ownership changed during collection")
                    if apply:
                        shutil.rmtree(path)
                        (root / ".usage" / f"{path.name}.json").unlink(missing_ok=True)
                    removed.append(row)
                    total -= row["total_bytes"]
            except TimeoutError:
                skipped.append(dict(cache_path=str(path), reason="active lease"))
        return dict(apply=apply, candidates=removed, skipped=skipped, remaining_bytes=total,
                    budget_met=max_bytes is None or total <= max_bytes)
