from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import stat
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from .runtime import read_json
from .memory import memory_info, check_memory


DEFAULT_EXCLUDES = (
    ".git/*",
    "*/.git/*",
    "__pycache__/*",
    "*/__pycache__/*",
    ".cache/*",
    "*/.cache/*",
    ".iloader-cache/*",
    "*/.iloader-cache/*",
    ".iloader/*",
    "*/.iloader/*",
    "*.lock",
    "*.tmp",
    "*.log",
)

ProgressFn = Callable[[int, int, Path], None]


@dataclass
class SourceFile:
    path: Path
    rel: Path
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int = 0
    device: int = 0
    inode: int = 0

    @property
    def rel_posix(self) -> str:
        return self.rel.as_posix()


@dataclass
class CachedFile:
    source: SourceFile
    fd: int
    digest_prefix: str

    @property
    def proc_path(self) -> str:
        return f"/proc/{os.getpid()}/fd/{self.fd}"

    def close(self) -> None:
        fd, self.fd = self.fd, -1
        try:
            os.close(fd)
        except OSError:
            pass


@dataclass
class CachedModel:
    source_root: Path
    files: list[CachedFile]
    total_bytes: int
    loaded_at: float = field(default_factory=time.time)
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDES
    workers: int = 1

    def close(self) -> None:
        for item in self.files:
            item.close()

    def to_manifest(self, view_path: Path | None = None) -> dict:
        return {
            "format_version": 1,
            "daemon_pid": os.getpid(),
            "source_root": str(self.source_root),
            "view_path": str(view_path) if view_path else None,
            "loaded_at": self.loaded_at,
            "file_count": len(self.files),
            "total_bytes": self.total_bytes,
            "include_patterns": list(self.include_patterns),
            "exclude_patterns": list(self.exclude_patterns),
            "workers": self.workers,
            "files": [
                {
                    "relative_path": item.source.rel_posix,
                    "size": item.source.size,
                    "source_mtime_ns": item.source.mtime_ns,
                    "memfd": item.proc_path,
                    "digest_prefix": item.digest_prefix,
                }
                for item in self.files
            ],
        }


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def available_memory_bytes() -> int | None:
    return memory_info()["available_bytes"]


def _matches_any(rel: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(Path(rel).name, pattern) for pattern in patterns)


def should_include(rel: Path, include_patterns: tuple[str, ...], exclude_patterns: tuple[str, ...]) -> bool:
    rel_text = rel.as_posix()
    if include_patterns and not _matches_any(rel_text, include_patterns):
        return False
    if exclude_patterns and _matches_any(rel_text, exclude_patterns):
        return False
    return True


def collect_source_files(
    source_root: Path,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDES,
) -> list[SourceFile]:
    source_root = source_root.expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not source_root.is_dir():
        raise NotADirectoryError(f"expected a model directory, got {source_root}")

    files: list[SourceFile] = []
    for path in sorted(source_root.rglob("*")):
        rel = path.relative_to(source_root)
        if not should_include(rel, include_patterns, exclude_patterns):
            continue
        st = path.stat()
        if not stat.S_ISREG(st.st_mode):
            continue
        files.append(SourceFile(path=path, rel=rel, size=st.st_size, mode=st.st_mode,
                                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns,
                                device=st.st_dev, inode=st.st_ino))
    return files


def inspect_source(
    source_root: Path,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDES,
) -> dict:
    files = collect_source_files(source_root, include_patterns, exclude_patterns)
    total = sum(item.size for item in files)
    by_suffix: dict[str, dict[str, int]] = {}
    for item in files:
        suffix = item.path.suffix or "<none>"
        bucket = by_suffix.setdefault(suffix, {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += item.size
    return {
        "source_root": str(Path(source_root).expanduser().resolve()),
        "file_count": len(files),
        "total_bytes": total,
        "total_human": human_bytes(total),
        "available_memory_bytes": available_memory_bytes(),
        "by_suffix": by_suffix,
    }


def source_identity(source_root: Path, include_patterns: tuple[str, ...] = (),
                    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDES) -> dict:
    root = source_root.expanduser().resolve()
    files = collect_source_files(root, include_patterns, exclude_patterns)
    description = {
        "source_root": str(root), "include_patterns": list(include_patterns),
        "exclude_patterns": list(exclude_patterns),
        "files": [(f.rel_posix, f.size, f.mtime_ns, f.ctime_ns, f.device, f.inode) for f in files],
    }
    return {"source_root": str(root), "fingerprint": hashlib.sha256(
        json.dumps(description, sort_keys=True).encode()).hexdigest()}


def _memfd_name(rel: Path) -> str:
    rel_text = rel.as_posix()
    digest = hashlib.sha1(rel_text.encode("utf-8")).hexdigest()[:12]
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in rel.name)
    return f"infiniloader-{digest}-{stem[:80]}"


def _copy_with_copy_file_range(src_fd: int, dst_fd: int, size: int, chunk_size: int) -> int:
    copy_file_range = getattr(os, "copy_file_range", None)
    if copy_file_range is None:
        raise OSError("copy_file_range is not available")
    copied = 0
    while copied < size:
        count = min(chunk_size, size - copied)
        sent = copy_file_range(src_fd, dst_fd, count)
        if sent == 0:
            break
        copied += sent
    return copied


def _copy_with_read_write(src_fd: int, dst_fd: int, size: int, chunk_size: int) -> int:
    copied = 0
    while copied < size:
        data = os.read(src_fd, min(chunk_size, max(size - copied, 1)))
        if not data:
            break
        view = memoryview(data)
        while view:
            written = os.write(dst_fd, view)
            view = view[written:]
        copied += len(data)
        del view, data
    return copied


def copy_file_to_memfd(source: SourceFile, chunk_size: int = 64 * 1024 * 1024) -> CachedFile:
    if not hasattr(os, "memfd_create"):
        raise RuntimeError("memfd_create is not available; InfiniLoader requires Linux")

    import fcntl

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    src_fd = os.open(source.path, os.O_RDONLY)
    dst_fd = None
    try:
        before = os.fstat(src_fd)
        if (before.st_size, before.st_mtime_ns) != (source.size, source.mtime_ns):
            raise RuntimeError(f"source changed before preload: {source.path}")
        if source.inode and (before.st_dev, before.st_ino, before.st_ctime_ns) != (source.device, source.inode, source.ctime_ns):
            raise RuntimeError(f"source replaced before preload: {source.path}")
        flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        dst_fd = os.memfd_create(_memfd_name(source.rel), flags)
        copied = 0
        try:
            copied = _copy_with_copy_file_range(src_fd, dst_fd, source.size, chunk_size)
        except OSError:
            os.lseek(src_fd, 0, os.SEEK_SET)
            os.lseek(dst_fd, 0, os.SEEK_SET)
            copied = 0

        if copied != source.size:
            os.ftruncate(dst_fd, 0)
            os.lseek(src_fd, 0, os.SEEK_SET)
            os.lseek(dst_fd, 0, os.SEEK_SET)
            copied = _copy_with_read_write(src_fd, dst_fd, source.size, chunk_size)

        if copied != source.size:
            raise IOError(f"short copy for {source.path}: {copied} != {source.size}")
        after = os.fstat(src_fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError(f"source changed during preload: {source.path}")

        os.lseek(dst_fd, 0, os.SEEK_SET)
        # Hash only a small prefix for diagnostics; full hashing doubles load time.
        digest = hashlib.sha256()
        remaining = min(source.size, 1024 * 1024)
        while remaining:
            data = os.read(dst_fd, min(chunk_size, remaining))
            if not data:
                raise IOError("short read while hashing cached prefix")
            digest.update(data)
            remaining -= len(data)
            del data
        os.lseek(dst_fd, 0, os.SEEK_SET)
        os.ftruncate(dst_fd, source.size)
        fcntl.fcntl(dst_fd, fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        os.fchmod(dst_fd, 0o400)
        return CachedFile(source=source, fd=dst_fd, digest_prefix=digest.hexdigest()[:16])
    except Exception:
        try:
            if dst_fd is not None:
                os.close(dst_fd)
        finally:
            raise
    finally:
        os.close(src_fd)


def load_model_into_memory(
    source_root: Path,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDES,
    max_memory_fraction: float = 0.90,
    force: bool = False,
    progress: ProgressFn | None = None,
    chunk_size: int = 64 * 1024 * 1024,
    workers: int = 1,
    max_memory_bytes: int | None = None,
    reserve_bytes: int = 256 * 1024 * 1024,
) -> CachedModel:
    if chunk_size <= 0 or workers < 1 or not 0 < max_memory_fraction <= 1:
        raise ValueError("chunk_size and workers must be positive; memory fraction must be in (0, 1]")
    source_root = source_root.expanduser().resolve()
    source_files = collect_source_files(source_root, include_patterns, exclude_patterns)
    total = sum(item.size for item in source_files)
    if not source_files:
        raise RuntimeError(f"no regular files selected under {source_root}")
    if progress:
        progress(0, total, Path("."))

    buffers = min(workers, len(source_files)) * min(chunk_size, max(f.size for f in source_files))
    check_memory(total + buffers, fraction=max_memory_fraction, max_bytes=max_memory_bytes,
                 reserve_bytes=reserve_bytes, force=force)

    copied = 0
    cached_by_index: list[CachedFile | None] = [None] * len(source_files)
    futures = {}
    try:
        if workers <= 1:
            for index, item in enumerate(source_files):
                cached_item = copy_file_to_memfd(item, chunk_size=chunk_size)
                cached_by_index[index] = cached_item
                copied += item.size
                if progress:
                    progress(copied, total, item.rel)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(copy_file_to_memfd, item, chunk_size): (index, item)
                    for index, item in enumerate(source_files)
                }
                for future in as_completed(futures):
                    index, item = futures[future]
                    cached_item = future.result()
                    cached_by_index[index] = cached_item
                    copied += item.size
                    if progress:
                        progress(copied, total, item.rel)
        if collect_source_files(source_root, include_patterns, exclude_patterns) != source_files:
            raise RuntimeError("source model changed during preload; retry using an immutable model revision")
        cached = [item for item in cached_by_index if item is not None]
        return CachedModel(
            source_root=source_root,
            files=cached,
            total_bytes=total,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            workers=max(1, workers),
        )
    except Exception:
        closed_fds = set()
        for item in cached_by_index:
            if item is None:
                continue
            item.close()
            closed_fds.add(item.fd)
        for future in futures:
            if not future.done() or future.cancelled():
                continue
            try:
                item = future.result()
            except Exception:
                continue
            if item.fd not in closed_fds:
                item.close()
                closed_fds.add(item.fd)
        raise


def _safe_remove_view(path: Path, expected_pid: int | None = None) -> None:
    if not path.exists() and not path.is_symlink():
        return
    resolved = path.resolve(strict=False)
    if str(resolved) in {"/", str(Path.home())}:
        raise RuntimeError(f"refusing to remove unsafe view path {path}")
    manifest = read_json(path / ".iloader" / "manifest.json") if path.is_dir() else None
    if path.is_symlink() or not manifest or manifest.get("format_version") != 1:
        raise RuntimeError(f"refusing to remove non-InfiniLoader directory {path}")
    if expected_pid is not None and manifest.get("daemon_pid") != expected_pid:
        raise RuntimeError(f"view belongs to another daemon: {path}")
    shutil.rmtree(path)


def build_view(cached: CachedModel, view_path: Path, replace: bool = False) -> Path:
    view_path = view_path.expanduser().absolute()
    if view_path.is_symlink():
        raise RuntimeError("view path must not be a symlink")
    if view_path.resolve().is_relative_to(cached.source_root) or cached.source_root.is_relative_to(view_path.resolve()):
        raise RuntimeError("view and source model directories must not overlap")
    parent = view_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(tempfile.mkdtemp(prefix=f".{view_path.name}.tmp.", dir=parent))

    try:
        for item in cached.files:
            dest = tmp_path / item.source.rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(item.proc_path, dest)

        manifest_dir = tmp_path / ".iloader"
        manifest_dir.mkdir(exist_ok=True)
        manifest = cached.to_manifest(view_path=view_path)
        (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        if view_path.exists() or view_path.is_symlink():
            if not replace:
                raise FileExistsError(f"{view_path} already exists; pass --replace")
            _safe_remove_view(view_path)
        os.replace(tmp_path, view_path)
        return view_path
    except Exception:
        shutil.rmtree(tmp_path)
        raise
