from __future__ import annotations

import os
import time
import statistics
from pathlib import Path

from .cache import DEFAULT_EXCLUDES, collect_source_files, human_bytes


def read_all_bytes(path: Path, chunk_size: int = 64 * 1024 * 1024) -> int:
    total = 0
    with path.open("rb", buffering=0) as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            total += len(data)
    return total


def benchmark_path(
    root: Path,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDES,
    repeat: int = 3,
    chunk_size: int = 64 * 1024 * 1024,
) -> dict:
    if repeat < 1 or chunk_size < 1:
        raise ValueError("repeat and chunk_size must be positive")
    root = root.expanduser().resolve()
    files = collect_source_files(root, include_patterns, exclude_patterns)
    total_size = sum(item.size for item in files)
    runs = []
    for idx in range(repeat):
        start = time.perf_counter()
        read_size = 0
        for item in files:
            read_size += read_all_bytes(root / item.rel, chunk_size=chunk_size)
        elapsed = time.perf_counter() - start
        if read_size != total_size:
            raise IOError(f"read {read_size} bytes, expected {total_size}")
        gib = read_size / (1024**3)
        runs.append(
            {
                "run": idx + 1,
                "bytes": read_size,
                "seconds": elapsed,
                "gib_per_second": gib / elapsed if elapsed > 0 else 0.0,
            }
        )
    best = max((run["gib_per_second"] for run in runs), default=0.0)
    return {
        "path": str(root),
        "file_count": len(files),
        "total_bytes": total_size,
        "total_human": human_bytes(total_size),
        "chunk_size": chunk_size,
        "repeat": repeat,
        "runs": runs,
        "best_gib_per_second": best,
        "median_seconds": statistics.median(run["seconds"] for run in runs),
        "first_run_seconds": runs[0]["seconds"],
        "cache_state": "uncontrolled; this benchmark does not clear OS page cache",
        "pid": os.getpid(),
    }
