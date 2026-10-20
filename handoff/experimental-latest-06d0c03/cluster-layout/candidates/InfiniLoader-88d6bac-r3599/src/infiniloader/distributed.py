from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import struct
import time
import queue as queue_module
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import DEFAULT_EXCLUDES, collect_source_files, human_bytes
from .paths import dist_view_root
from .runtime import validate_name, read_json


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


@dataclass
class TensorInfo:
    file: str
    name: str
    dtype: str
    shape: list[int]
    data_offsets: list[int]

    @property
    def nbytes(self) -> int:
        if self.data_offsets and len(self.data_offsets) == 2:
            return int(self.data_offsets[1]) - int(self.data_offsets[0])
        elems = 1
        for dim in self.shape:
            elems *= dim
        return elems * DTYPE_BYTES.get(self.dtype, 0)


def parse_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is too small to be safetensors")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len > 100 * 1024 * 1024 or header_len > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = f.read(header_len)
    return json.loads(header.decode("utf-8"))


def collect_safetensors_metadata(model_dir: Path) -> list[TensorInfo]:
    model_dir = model_dir.expanduser().resolve()
    rows: list[TensorInfo] = []
    for file_path in sorted(model_dir.glob("*.safetensors")):
        header = parse_safetensors_header(file_path)
        for name, item in header.items():
            if name == "__metadata__":
                continue
            rows.append(
                TensorInfo(
                    file=file_path.relative_to(model_dir).as_posix(),
                    name=name,
                    dtype=item.get("dtype", ""),
                    shape=list(item.get("shape", [])),
                    data_offsets=list(item.get("data_offsets", [])),
                )
            )
    return rows


def infer_tp_split(tensor: TensorInfo, tp: int) -> dict[str, Any]:
    name = tensor.name
    shape = tensor.shape
    lower = name.lower()
    result = {
        "name": name,
        "file": tensor.file,
        "dtype": tensor.dtype,
        "shape": shape,
        "bytes": tensor.nbytes,
        "bytes_human": human_bytes(tensor.nbytes),
        "action": "replicate",
        "dim": None,
        "reason": "not recognized as a tensor-parallel weight",
        "divisible": True,
    }

    if len(shape) < 1:
        return result

    # Heuristics only. Framework adapters must confirm exact naming conventions.
    col_parallel = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "query_key_value.weight",
        "dense_h_to_4h.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "w1.weight",
        "w3.weight",
        "lm_head.weight",
        "embed_tokens.weight",
        "word_embeddings.weight",
    )
    row_parallel = (
        "o_proj.weight",
        "dense.weight",
        "dense_4h_to_h.weight",
        "down_proj.weight",
        "w2.weight",
    )
    if any(lower.endswith(suffix) or suffix in lower for suffix in col_parallel):
        dim = 0
        action = "split"
        reason = "column-parallel style tensor; split output/vocab dimension"
    elif any(lower.endswith(suffix) or suffix in lower for suffix in row_parallel):
        dim = 1 if len(shape) >= 2 else 0
        action = "split"
        reason = "row-parallel style tensor; split input dimension"
    else:
        dim = None
        action = "replicate"
        reason = "bias, norm, rotary, scalar, or unknown tensor; replicate by default"

    result["action"] = action
    result["dim"] = dim
    result["reason"] = reason
    if action == "split" and dim is not None:
        result["divisible"] = shape[dim] % tp == 0
        result["per_rank_shape"] = [
            (value // tp if index == dim else value) for index, value in enumerate(shape)
        ]
        result["per_rank_bytes_human"] = human_bytes(tensor.nbytes // tp if result["divisible"] else tensor.nbytes)
    return result


def build_tp_plan(model_dir: Path, tp: int) -> dict[str, Any]:
    if tp < 1:
        raise ValueError("tp must be positive")
    tensors = collect_safetensors_metadata(model_dir)
    planned = [infer_tp_split(tensor, tp) for tensor in tensors]
    split = [item for item in planned if item["action"] == "split"]
    nondiv = [item for item in split if not item["divisible"]]
    total_bytes = sum(item["bytes"] for item in planned)
    split_bytes = sum(item["bytes"] for item in split if item["divisible"])
    return {
        "model_dir": str(Path(model_dir).expanduser().resolve()),
        "tp": tp,
        "tensor_count": len(planned),
        "total_bytes": total_bytes,
        "total_human": human_bytes(total_bytes),
        "split_tensor_count": len(split),
        "split_bytes": split_bytes,
        "split_human": human_bytes(split_bytes),
        "nondivisible_count": len(nondiv),
        "compatible_without_layout_change": True,
        "can_emit_rank_sharded_hf_dir_without_adapter": False,
        "note": (
            "This is an analysis plan. The default distributed-safe path is a full HF-compatible "
            "view per rank. Emitting rank-sharded weight files requires a framework-specific loader "
            "or an engine-supported sharded layout."
        ),
        "tensors": planned,
    }


def _safe_remove_dist_view(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    marker = read_json(path / ".iloader-dist" / "manifest.json")
    if path.is_symlink() or not marker or marker.get("mode") != "full-hf-compatible":
        raise RuntimeError(f"refusing to remove non-InfiniLoader distributed view {path}")
    shutil.rmtree(path)


def _rank_view_target(source: Path, rel: Path) -> str:
    src = source / rel
    if src.is_symlink():
        # Keep the daemon's /proc/<pid>/fd symlink chain reopenable from rank views.
        return str(src.absolute())
    return str(src.resolve())


def create_rank_views(
    source: Path,
    *,
    name: str,
    ranks: int,
    base_dir: str | os.PathLike[str] | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    validate_name(name)
    if ranks < 1:
        raise ValueError("ranks must be positive")
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    root = dist_view_root(base_dir) / name
    if root.resolve().is_relative_to(source) or source.is_relative_to(root.resolve()):
        raise RuntimeError("distributed view and source directories must not overlap")
    files = collect_source_files(source, (), DEFAULT_EXCLUDES)
    tmp = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp.", dir=root.parent))
    try:
        for rank in range(ranks):
            rank_dir = tmp / f"rank{rank}"
            rank_dir.mkdir(parents=True)
            for item in files:
                dest = rank_dir / item.rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(_rank_view_target(source, item.rel), dest)
            meta = {
                "rank": rank,
                "ranks": ranks,
                "source": str(source),
                "mode": "full-hf-compatible",
                "file_count": len(files),
                "total_bytes": sum(item.size for item in files),
            }
            marker = rank_dir / ".iloader-dist"
            marker.mkdir(exist_ok=True)
            (marker / "rank.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        root_marker = tmp / ".iloader-dist"
        root_marker.mkdir(exist_ok=True)
        manifest = {
            "name": name,
            "source": str(source),
            "ranks": ranks,
            "mode": "full-hf-compatible",
            "rank_paths": [str((root / f"rank{rank}").resolve(strict=False)) for rank in range(ranks)],
            "file_count": len(files),
            "total_bytes": sum(item.size for item in files),
            "total_human": human_bytes(sum(item.size for item in files)),
        }
        (root_marker / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        if root.exists() or root.is_symlink():
            if not replace:
                raise FileExistsError(f"{root} already exists; pass --replace")
            _safe_remove_dist_view(root)
        os.replace(tmp, root)
        manifest["rank_paths"] = [str((root / f"rank{rank}").resolve()) for rank in range(ranks)]
        return manifest
    except Exception:
        shutil.rmtree(tmp)
        raise


def _read_files_worker(rank: int, root: str, include_patterns: tuple[str, ...], exclude_patterns: tuple[str, ...], barrier: Any, queue: Any, chunk_size: int) -> None:
    try:
        path = Path(root)
        files = collect_source_files(path, include_patterns, exclude_patterns)
        total = sum(item.size for item in files)
        barrier.wait(timeout=30)
        start = time.perf_counter()
        read_bytes = 0
        for item in files:
            with (path / item.rel).open("rb", buffering=0) as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    read_bytes += len(data)
        elapsed = time.perf_counter() - start
        if read_bytes != total:
            raise IOError(f"read {read_bytes} bytes, expected {total}")
        queue.put({"rank": rank, "path": root, "bytes": read_bytes, "expected_bytes": total, "seconds": elapsed, "ok": True})
    except Exception as exc:
        barrier.abort()
        queue.put({"rank": rank, "path": root, "ok": False, "error": str(exc)})


def simulate_distributed_read(
    path: Path,
    *,
    ranks: int,
    rank_view_root: Path | None = None,
    include_patterns: tuple[str, ...] = ("*.safetensors",),
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDES,
    chunk_size: int = 64 * 1024 * 1024,
    timeout: float = 60.0,
) -> dict[str, Any]:
    if ranks < 1 or chunk_size <= 0 or timeout <= 0:
        raise ValueError("ranks, chunk_size and timeout must be positive")
    roots = []
    if rank_view_root:
        for rank in range(ranks):
            roots.append(str((rank_view_root / f"rank{rank}").expanduser().resolve()))
    else:
        root = str(path.expanduser().resolve())
        roots = [root for _ in range(ranks)]

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(ranks)
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_read_files_worker, args=(rank, roots[rank], include_patterns, exclude_patterns, barrier, queue, chunk_size))
        for rank in range(ranks)
    ]
    start = time.perf_counter()
    rows = []
    try:
        for proc in processes:
            proc.start()
        deadline = time.monotonic() + timeout
        while len(rows) < ranks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                row = queue.get(timeout=min(remaining, .1))
                rows.append(row)
                if not row.get("ok"):
                    break
            except queue_module.Empty:
                if any(proc.exitcode not in (None, 0) for proc in processes):
                    break
        received = {row["rank"] for row in rows}
        for rank in range(ranks):
            if rank not in received:
                rows.append({"rank": rank, "path": roots[rank], "ok": False,
                             "error": "rank failed, timed out or was cancelled"})
    finally:
        barrier.abort()
        for proc in processes:
            if proc.pid is None:
                continue
            proc.join(timeout=.2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)
        queue.close()
        queue.cancel_join_thread()
    elapsed = time.perf_counter() - start
    ok = all(row.get("ok") for row in rows)
    aggregate_bytes = sum(row.get("bytes", 0) for row in rows)
    return {
        "ok": ok,
        "ranks": ranks,
        "paths": roots,
        "wall_seconds": elapsed,
        "aggregate_bytes": aggregate_bytes,
        "aggregate_human": human_bytes(aggregate_bytes),
        "aggregate_gib_per_second": (aggregate_bytes / (1024**3) / elapsed) if elapsed > 0 else 0.0,
        "rank_results": sorted(rows, key=lambda row: row.get("rank", -1)),
    }
