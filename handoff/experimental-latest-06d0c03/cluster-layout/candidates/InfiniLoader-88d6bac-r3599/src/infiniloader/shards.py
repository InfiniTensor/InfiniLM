"""Explicit, portable tensor layouts. No model-name based partition inference.

The output is an InfiniLoader checkpoint, not a drop-in sharded HF checkpoint.
An engine must implement the supplied TP/PP contract (see the torch example).
"""
from __future__ import annotations

import hashlib
import ctypes
import errno
import json
import math
import os
import shutil
import stat
import struct
import tempfile
from pathlib import Path

from .distributed import DTYPE_BYTES

CHUNK = 1024 * 1024
MAX_HEADER = 100_000_000


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def _signature(path: Path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _int(value, *, minimum=0):
    return type(value) is int and value >= minimum


def tensor_header(path: Path) -> tuple[dict, int]:
    """Reject holes, overlaps, truncated payloads and unsupported packed dtypes."""
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors: {path}")
        length, = struct.unpack("<Q", raw)
        if not 2 <= length <= min(MAX_HEADER, size - 8):
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(stream.read(length).decode("utf-8"), object_pairs_hook=_unique)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("safetensors metadata must contain strings")
    intervals = []
    for name, spec in header.items():
        if not name or not isinstance(spec, dict):
            raise ValueError("invalid tensor record")
        dtype, shape, offsets = spec.get("dtype"), spec.get("shape"), spec.get("data_offsets")
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or not all(_int(d) for d in shape):
            raise ValueError(f"unsupported dtype or shape: {name}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(_int(v) for v in offsets) or offsets[1] < offsets[0]
                or offsets[1] - offsets[0] != math.prod(shape) * DTYPE_BYTES[dtype]):
            raise ValueError(f"invalid tensor offsets/size: {name}")
        intervals.append(tuple(offsets))
    end = 0
    for start, stop in sorted(intervals):
        if start != end:
            raise ValueError("overlapping tensor data or hole in payload")
        end = stop
    if end != size - 8 - length:
        raise ValueError("truncated or trailing safetensors payload")
    return header, 8 + length


def catalog(source: Path) -> dict:
    """Content identity is independent of hostname, inode and source directory."""
    source = source.expanduser().resolve()
    files, tensors = {}, {}
    paths = sorted(source.rglob("*.safetensors"))
    signatures = {p: _signature(p) for p in paths}
    for path in paths:
        before = signatures[path]
        header, base = tensor_header(path)
        rel = path.relative_to(source).as_posix()
        files[rel] = {"sha256": file_hash(path), "size": before[2], "data_start": base}
        if before != _signature(path):
            raise RuntimeError(f"source changed while cataloging: {path}")
        for name, spec in header.items():
            if name in tensors:
                raise ValueError(f"duplicate tensor across files: {name}")
            tensors[name] = dict(spec, file=rel)
    if not files or not tensors:
        raise ValueError("no safetensors tensors found")
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        if not isinstance(index, dict) or index.get("weight_map") != {n: t["file"] for n, t in tensors.items()}:
            raise ValueError("safetensors index does not match tensor inventory")
    if paths != sorted(source.rglob("*.safetensors")) or any(_signature(p) != s for p, s in signatures.items()):
        raise RuntimeError("source inventory changed while cataloging")
    result = {"format": 1, "files": files, "tensors": tensors}
    result["content_id"] = digest(result)
    result["tensor_bytes"] = sum(t["data_offsets"][1] - t["data_offsets"][0] for t in tensors.values())
    return result


def build_plan(source: Path, layout: dict, *, mode="sharded", capacities: list[int] | None = None) -> dict:
    cat = catalog(source)
    if not isinstance(layout, dict) or layout.get("format") != 1:
        raise ValueError("layout requires format=1")
    tp, pp = layout.get("tp"), layout.get("pp")
    rules = layout.get("tensors")
    if not _int(tp, minimum=1) or not _int(pp, minimum=1) or not isinstance(rules, dict):
        raise ValueError("layout requires positive tp, pp and explicit tensor rules")
    if mode not in ("sharded", "replicated") or set(rules) != set(cat["tensors"]):
        raise ValueError("invalid mode or layout must explicitly cover exactly all tensors")
    for name, spec in cat["tensors"].items():
        rule = rules[name]
        if not isinstance(rule, dict) or set(rule) != {"stage", "split_dim"}:
            raise ValueError(f"rule requires stage and split_dim: {name}")
        stage, dim = rule["stage"], rule["split_dim"]
        if not _int(stage) or stage >= pp:
            raise ValueError(f"invalid pipeline stage: {name}")
        if dim is not None and (not _int(dim) or dim >= len(spec["shape"]) or spec["shape"][dim] % tp):
            raise ValueError(f"invalid or nondivisible split dimension: {name}")
    ranks = []
    for rank in range(tp * pp):
        stage, lane = divmod(rank, tp)
        parts = {}
        for name, spec in cat["tensors"].items():
            rule = rules[name]
            if mode == "sharded" and rule["stage"] != stage:
                continue
            dim = rule["split_dim"] if mode == "sharded" else None
            shape = list(spec["shape"])
            start, stop = 0, 0
            if dim is not None:
                width = shape[dim] // tp
                start, stop, shape[dim] = lane * width, (lane + 1) * width, width
            parts[name] = {"dim": dim, "start": start, "stop": stop, "shape": shape, "dtype": spec["dtype"],
                           "bytes": math.prod(shape) * DTYPE_BYTES[spec["dtype"]]}
        ranks.append({"rank": rank, "stage": stage, "lane": lane, "tensors": parts,
                      "tensor_bytes": sum(p["bytes"] for p in parts.values())})
    if any(not r["tensors"] for r in ranks):
        raise ValueError("every pipeline stage must own at least one tensor")
    if capacities is not None:
        if len(capacities) != len(ranks) or not all(_int(v) for v in capacities):
            raise ValueError("capacities requires one nonnegative byte limit per rank")
        if any(r["tensor_bytes"] > c for r, c in zip(ranks, capacities)):
            raise MemoryError("rank tensor payload exceeds node capacity (headers/buffers are additional)")
    plan = {"format": 1, "content_id": cat["content_id"], "mode": mode, "layout": layout,
            "unique_tensor_bytes": cat["tensor_bytes"], "ranks": ranks,
            "aggregate_tensor_bytes": sum(r["tensor_bytes"] for r in ranks)}
    plan["plan_id"] = digest(plan)
    return plan


def iter_part(source: Path, cat: dict, name: str, part: dict):
    if not part["bytes"]:
        return
    spec = cat["tensors"][name]
    begin = cat["files"][spec["file"]]["data_start"] + spec["data_offsets"][0]
    dim = part["dim"]
    if dim is None:
        rows, stride, offset, width = 1, 0, 0, part["bytes"]
    else:
        shape = spec["shape"]
        inner = math.prod(shape[dim + 1:]) * DTYPE_BYTES[spec["dtype"]]
        rows, stride = math.prod(shape[:dim]), shape[dim] * inner
        offset, width = part["start"] * inner, (part["stop"] - part["start"]) * inner
    with (source / spec["file"]).open("rb") as stream:
        for row in range(rows):
            stream.seek(begin + row * stride + offset)
            left = width
            while left:
                block = stream.read(min(left, CHUNK))
                if not block:
                    raise IOError("source truncated during shard export")
                left -= len(block)
                yield block


def _publish_directory(temp: Path, destination: Path):
    if os.name == "nt":
        os.rename(temp, destination)  # Windows rename never replaces a destination.
        return
    # POSIX rename would replace a concurrently created empty directory. Some NFS
    # servers lack RENAME_NOREPLACE: publish a relative symlink to the complete,
    # private staging directory instead. symlink creation is atomic and exclusive.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        os.symlink(temp.name, destination, target_is_directory=True)
        return
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(temp), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            os.symlink(temp.name, destination, target_is_directory=True)
            return
        raise OSError(code, os.strerror(code), destination)


def export_rank(source: Path, plan: dict, rank: int, destination: Path) -> dict:
    """Stream slices with bounded buffers; publish a new directory atomically."""
    source, destination = source.expanduser().resolve(), destination.expanduser().absolute()
    if destination.is_symlink() or destination.exists():
        raise FileExistsError(destination)
    resolved = destination.resolve()
    if resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise ValueError("export and source directories must not overlap")
    expected = build_plan(source, plan["layout"], mode=plan["mode"])
    if plan != expected:
        raise ValueError("stale or modified shard plan")
    if not _int(rank) or rank >= len(plan["ranks"]):
        raise ValueError("invalid rank")
    cat = catalog(source)
    if cat["content_id"] != plan["content_id"]:
        raise ValueError("source changed since plan validation")
    signatures = {p: _signature(source / p) for p in cat["files"]}
    selected = plan["ranks"][rank]
    header, offset = {}, 0
    for name, part in selected["tensors"].items():
        header[name] = {"dtype": cat["tensors"][name]["dtype"], "shape": part["shape"],
                        "data_offsets": [offset, offset + part["bytes"]]}
        offset += part["bytes"]
    raw = canonical(header)
    raw += b" " * (-len(raw) % 8)
    if len(raw) > MAX_HEADER:
        raise ValueError("exported safetensors header exceeds format limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".iloader-export-", dir=destination.parent))
    try:
        weights = temp / "weights.safetensors"
        with weights.open("wb") as stream:
            stream.write(struct.pack("<Q", len(raw)))
            stream.write(raw)
            for name, part in selected["tensors"].items():
                for block in iter_part(source, cat, name, part):
                    stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if any(_signature(source / p) != s for p, s in signatures.items()):
            raise RuntimeError("source changed during shard export")
        if catalog(source)["content_id"] != plan["content_id"]:
            raise RuntimeError("source content changed during shard export")
        receipt = {"format": 1, "kind": "infiniloader-explicit-shard", "plan": plan,
                   "rank": rank, "weights_sha256": file_hash(weights)}
        (temp / "shard.json").write_bytes(canonical(receipt))
        _publish_directory(temp, destination)
        return receipt
    except BaseException:
        # A signal can arrive just after successful symlink publication. Never
        # remove a backing directory now reachable through the public rank path.
        try:
            published = (stat.S_ISLNK(destination.lstat().st_mode)
                         and destination.readlink() == Path(temp.name))
        except (FileNotFoundError, NotADirectoryError):
            published = False
        except OSError:
            published = True  # Unknown publication result: preserve data.
        if not published:
            shutil.rmtree(temp, ignore_errors=True)
        raise


def verify_rank(path: Path) -> dict:
    receipt = read_json(path / "shard.json")
    if (not isinstance(receipt, dict) or receipt.get("format") != 1
            or receipt.get("kind") != "infiniloader-explicit-shard"):
        raise ValueError("invalid InfiniLoader shard receipt")
    plan = receipt["plan"]
    if plan.get("plan_id") != digest({k: v for k, v in plan.items() if k != "plan_id"}):
        raise ValueError("invalid plan identity")
    rank = receipt["rank"]
    if not _int(rank) or rank >= len(plan["ranks"]):
        raise ValueError("invalid shard rank")
    if file_hash(path / "weights.safetensors") != receipt["weights_sha256"]:
        raise ValueError("shard checksum mismatch")
    header, _ = tensor_header(path / "weights.safetensors")
    parts = plan["ranks"][rank]["tensors"]
    if set(header) != set(parts) or any(t["shape"] != parts[n]["shape"] or t["dtype"] != parts[n]["dtype"]
                                     for n, t in header.items()):
        raise ValueError("shard tensor inventory/layout mismatch")
    return receipt


def configure_parser(sub):
    parser = sub.add_parser("shard", help="explicit tensor layout planning, export and verification")
    actions = parser.add_subparsers(dest="shard_action", required=True)
    plan = actions.add_parser("plan")
    plan.add_argument("source", type=Path)
    plan.add_argument("--layout", type=Path, required=True)
    plan.add_argument("--mode", choices=("sharded", "replicated"), default="sharded")
    plan.add_argument("--output", type=Path, required=True)
    export = actions.add_parser("export")
    export.add_argument("source", type=Path)
    export.add_argument("--plan", type=Path, required=True)
    export.add_argument("--rank", type=int, required=True)
    export.add_argument("--output", type=Path, required=True)
    verify = actions.add_parser("verify")
    verify.add_argument("path", type=Path)

    def run(args):
        if args.shard_action == "plan":
            result = build_plan(args.source, read_json(args.layout), mode=args.mode)
            # Never silently replace an existing deployment plan.
            with args.output.open("xb") as stream:
                stream.write(canonical(result))
        elif args.shard_action == "export":
            result = export_rank(args.source, read_json(args.plan), args.rank, args.output)
        else:
            result = verify_rank(args.path)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    for command in (plan, export, verify):
        command.set_defaults(func=run)
