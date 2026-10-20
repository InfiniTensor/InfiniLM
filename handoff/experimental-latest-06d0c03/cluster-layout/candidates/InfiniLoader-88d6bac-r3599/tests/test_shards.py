from __future__ import annotations

import json
import shutil
import struct

import pytest

from infiniloader.shards import (build_plan, catalog, canonical, export_rank,
                                tensor_header, verify_rank)


def write_tensor(path, shape=(4, 6, 2), name="weight"):
    path.mkdir()
    values = list(range(__import__("math").prod(shape)))
    payload = struct.pack("<" + "f" * len(values), *values)
    header = canonical({name: {"dtype": "F32", "shape": list(shape), "data_offsets": [0, len(payload)]}})
    (path / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return payload


def layout(dim, tp=2):
    return {"format": 1, "tp": tp, "pp": 1, "tensors": {"weight": {"stage": 0, "split_dim": dim}}}


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_slices_reconstruct_every_axis(tmp_path, dim, monkeypatch):
    import infiniloader.shards as shards
    monkeypatch.setattr(shards, "CHUNK", 7)
    source = tmp_path / "source"
    original = write_tensor(source)
    plan = build_plan(source, layout(dim))
    assert plan["aggregate_tensor_bytes"] == len(original)
    outputs = []
    for rank in range(2):
        dest = tmp_path / f"rank{rank}"
        export_rank(source, plan, rank, dest)
        verify_rank(dest)
        header, base = tensor_header(dest / "weights.safetensors")
        outputs.append((dest / "weights.safetensors").read_bytes()[base:])
    shape = [4, 6, 2]
    inner = __import__("math").prod(shape[dim + 1:]) * 4
    width = shape[dim] // 2 * inner
    rows = __import__("math").prod(shape[:dim])
    rebuilt = b"".join(part[row * width:(row + 1) * width] for row in range(rows) for part in outputs)
    assert rebuilt == original


def test_replicas_and_portable_content_id(tmp_path):
    source = tmp_path / "source"
    payload = write_tensor(source)
    shutil.copytree(source, tmp_path / "relocated")
    assert catalog(source) == catalog(tmp_path / "relocated")
    plan = build_plan(source, layout(0), mode="replicated")
    assert plan["aggregate_tensor_bytes"] == 2 * len(payload)
    assert all(r["tensor_bytes"] == len(payload) for r in plan["ranks"])
    with pytest.raises(MemoryError):
        build_plan(source, layout(0), capacities=[1, 1])


def test_pipeline_has_only_assigned_tensors(tmp_path):
    source = tmp_path / "source"
    write_tensor(source, name="a")
    other = tmp_path / "other"
    write_tensor(other, name="b")
    shutil.copyfile(other / "weights.safetensors", source / "other.safetensors")
    rules = {"format": 1, "tp": 2, "pp": 2, "tensors": {
        "a": {"stage": 0, "split_dim": 0}, "b": {"stage": 1, "split_dim": 1}}}
    plan = build_plan(source, rules)
    assert plan["aggregate_tensor_bytes"] == catalog(source)["tensor_bytes"]
    assert set(plan["ranks"][3]["tensors"]) == {"b"}
    rules["tensors"]["b"]["split_dim"] = None
    assert build_plan(source, rules)["aggregate_tensor_bytes"] > plan["aggregate_tensor_bytes"]


@pytest.mark.parametrize("mutate", [
    lambda x: x["tensors"].clear(),
    lambda x: x["tensors"]["weight"].update(split_dim=3),
    lambda x: x.update(tp=5),
    lambda x: x.update(tp=True),
    lambda x: x["tensors"]["weight"].update(stage=1),
])
def test_reject_ambiguous_or_invalid_layout(tmp_path, mutate):
    source = tmp_path / "source"
    write_tensor(source)
    rules = layout(0)
    mutate(rules)
    with pytest.raises(ValueError):
        build_plan(source, rules)


def test_no_overwrite_stale_plan_and_corruption(tmp_path):
    source = tmp_path / "source"
    write_tensor(source)
    plan = build_plan(source, layout(0))
    dest = tmp_path / "out"
    export_rank(source, plan, 0, dest)
    with pytest.raises(FileExistsError):
        export_rank(source, plan, 0, dest)
    weights = dest / "weights.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="checksum"):
        verify_rank(dest)
    weights = source / "weights.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="stale"):
        export_rank(source, plan, 0, tmp_path / "other")


@pytest.mark.parametrize("header,payload", [
    (b'{"a":{},"a":{}}', b""),
    (canonical({"a": {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}}), b"1234"),
    (canonical({"a": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]}}), b"12345"),
    (canonical({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}), b"12345"),
    (canonical({"__metadata__": {"x": 42}}), b""),
    (canonical({"a": {"dtype": "F4", "shape": [2], "data_offsets": [0, 1]}}), b"x"),
])
def test_reject_invalid_safetensors(tmp_path, header, payload):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    with pytest.raises(ValueError):
        tensor_header(path)


def test_scalar_empty_duplicate_tensor_and_index(tmp_path):
    source = tmp_path / "source"
    write_tensor(source, shape=())
    assert catalog(source)["tensor_bytes"] == 4
    shutil.copy(source / "weights.safetensors", source / "duplicate.safetensors")
    with pytest.raises(ValueError, match="duplicate tensor"):
        catalog(source)
    (source / "duplicate.safetensors").unlink()
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    with pytest.raises(ValueError, match="index"):
        catalog(source)
    empty = tmp_path / "empty"
    write_tensor(empty, shape=(0, 2))
    assert catalog(empty)["tensor_bytes"] == 0


def test_catalog_detects_change_to_previously_hashed_file(tmp_path, monkeypatch):
    import infiniloader.shards as shards
    source = tmp_path / "source"
    write_tensor(source, name="a")
    other = tmp_path / "other"
    write_tensor(other, name="b")
    shutil.copyfile(other / "weights.safetensors", source / "z.safetensors")
    real = shards.file_hash
    def change_after(path):
        value = real(path)
        if path.name == "z.safetensors":
            target = source / "weights.safetensors"
            target.write_bytes(target.read_bytes()[:-4] + struct.pack("<f", 999))
        return value
    monkeypatch.setattr(shards, "file_hash", change_after)
    with pytest.raises(RuntimeError, match="changed"):
        shards.catalog(source)


def test_catalog_to_export_race_is_rejected(tmp_path, monkeypatch):
    import infiniloader.shards as shards
    source = tmp_path / "source"
    write_tensor(source)
    plan = build_plan(source, layout(0))
    real, calls = shards.catalog, 0
    def change_after(path):
        nonlocal calls
        value = real(path)
        calls += 1
        if calls == 2:
            target = source / "weights.safetensors"
            target.write_bytes(target.read_bytes()[:-4] + struct.pack("<f", 999))
        return value
    monkeypatch.setattr(shards, "catalog", change_after)
    with pytest.raises(RuntimeError, match="changed"):
        export_rank(source, plan, 1, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_atomic_publication_preserves_concurrent_empty_destination(tmp_path, monkeypatch):
    import infiniloader.shards as shards
    source, dest = tmp_path / "source", tmp_path / "out"
    write_tensor(source)
    plan = build_plan(source, layout(0))
    real = shards._publish_directory
    def concurrent(temp, destination):
        destination.mkdir()
        real(temp, destination)
    monkeypatch.setattr(shards, "_publish_directory", concurrent)
    with pytest.raises(FileExistsError):
        export_rank(source, plan, 0, dest)
    assert dest.is_dir() and not list(dest.iterdir())


def test_empty_huge_outer_dimension_exports_without_iteration(tmp_path):
    source = tmp_path / "source"
    write_tensor(source, shape=(10**9, 0))
    plan = build_plan(source, layout(1))
    export_rank(source, plan, 0, tmp_path / "out")
    assert verify_rank(tmp_path / "out")["rank"] == 0


def test_reject_utf16_header_and_interior_empty_offsets(tmp_path):
    file = tmp_path / "bad.safetensors"
    raw = '{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'.encode("utf-16")
    file.write_bytes(struct.pack("<Q", len(raw)) + raw + b"1234")
    with pytest.raises((ValueError, UnicodeError)):
        tensor_header(file)
    raw = canonical({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
                     "b": {"dtype": "F32", "shape": [0], "data_offsets": [8, 8]}})
    file.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(16))
    with pytest.raises(ValueError):
        tensor_header(file)


@pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX network filesystem fallback")
def test_network_fs_publication_is_atomic_and_exclusive(tmp_path, monkeypatch):
    import ctypes
    import errno
    from types import SimpleNamespace
    import infiniloader.shards as shards
    def unsupported(*args):
        ctypes.set_errno(errno.EOPNOTSUPP)
        return -1
    monkeypatch.setattr(shards.ctypes, "CDLL", lambda *a, **k: SimpleNamespace(renameat2=unsupported))
    source = tmp_path / "source"
    write_tensor(source)
    plan = build_plan(source, layout(0))
    dest = tmp_path / "out"
    export_rank(source, plan, 0, dest)
    assert dest.is_symlink()
    assert not dest.readlink().is_absolute()
    assert verify_rank(dest)["rank"] == 0
    # An existing empty directory must also survive on the fallback path.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        shards._publish_directory(dest.resolve(), empty)
    assert empty.is_dir() and not list(empty.iterdir())
    publish = shards._publish_directory
    def interrupted(temp, destination):
        publish(temp, destination)
        raise KeyboardInterrupt()
    monkeypatch.setattr(shards, "_publish_directory", interrupted)
    interrupted_dest = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        export_rank(source, plan, 1, interrupted_dest)
    assert interrupted_dest.is_symlink()
    assert verify_rank(interrupted_dest)["rank"] == 1


@pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX network filesystem fallback")
def test_uncertain_published_path_is_preserved(tmp_path, monkeypatch):
    import errno
    import os
    from pathlib import Path
    import infiniloader.shards as shards
    source = tmp_path / "source"
    write_tensor(source)
    plan = build_plan(source, layout(0))
    dest = tmp_path / "out"
    original_lstat = Path.lstat
    def io_error(path, *args, **kwargs):
        if path == dest:
            raise OSError(errno.EIO, "injected uncertain NFS lookup")
        return original_lstat(path, *args, **kwargs)
    def uncertain(temp, destination):
        os.symlink(temp.name, destination)
        monkeypatch.setattr(Path, "lstat", io_error)
        raise KeyboardInterrupt()
    monkeypatch.setattr(shards, "_publish_directory", uncertain)
    with pytest.raises(KeyboardInterrupt):
        export_rank(source, plan, 0, dest)
    monkeypatch.setattr(Path, "lstat", original_lstat)
    assert verify_rank(dest)["rank"] == 0
