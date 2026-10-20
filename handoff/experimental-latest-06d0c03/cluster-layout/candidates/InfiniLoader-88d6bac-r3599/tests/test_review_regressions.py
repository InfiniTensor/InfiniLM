"""Cache integrity and allocation regressions from independent expert review."""
import json
import os
import tracemalloc

import pytest

from conftest import make_model_dir
from infiniloader import cache
from infiniloader.local_cache import garbage_collect, materialize_model_cache, verify_cache
from infiniloader.runtime import atomic_json

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux review regressions")


def test_extra_model_file_invalidates_and_rebuilds_generation(tmp_path):
    model = make_model_dir(tmp_path / "model")
    first = materialize_model_cache(model, base_dir=tmp_path / "state")
    (first.cache_path / "unexpected.safetensors").write_bytes(b"extra weights")
    assert not verify_cache(first.cache_path, full=False)["valid"]
    assert not verify_cache(first.cache_path)["valid"]
    second = materialize_model_cache(model, base_dir=tmp_path / "state")
    assert not second.reused
    assert second.cache_path != first.cache_path
    assert not (second.cache_path / "unexpected.safetensors").exists()


@pytest.mark.parametrize("malformation", ["null", "duplicate", "invalid_source_name", "bad_created_at", "overflow_time", "wrong_total", "bool_total"])
def test_corrupt_manifest_is_reported_and_refresh_can_recover(tmp_path, malformation):
    model = make_model_dir(tmp_path / "model")
    first = materialize_model_cache(model, base_dir=tmp_path / "state")
    file = first.cache_path / ".iloader-cache/manifest.json"
    data = json.loads(file.read_text())
    if malformation == "null":
        data["cached_files"][0] = None
    elif malformation == "duplicate":
        data["cached_files"][0] = data["cached_files"][1]
    elif malformation == "invalid_source_name":
        data["files"][0]["relative_path"] = None
    elif malformation == "bad_created_at":
        data["created_at"] = {}
        data["cached_files"][0] = None
    elif malformation == "overflow_time":
        data["created_at"] = 10**1000
        data["cached_files"][0] = None
    else:
        data["total_bytes"] = False if malformation == "bool_total" else 0
    atomic_json(file, data)
    assert not verify_cache(first.cache_path)["valid"]
    second = materialize_model_cache(model, base_dir=tmp_path / "state", refresh=True)
    assert second.cache_path != first.cache_path
    assert verify_cache(second.cache_path)["valid"]


def test_gc_counts_files_even_if_manifest_total_is_wrong(tmp_path):
    local = materialize_model_cache(make_model_dir(tmp_path / "model"), base_dir=tmp_path / "state")
    file = local.cache_path / ".iloader-cache/manifest.json"
    data = json.loads(file.read_text())
    data["total_bytes"] = 0
    atomic_json(file, data)
    result = garbage_collect(base_dir=tmp_path / "state", max_bytes=0)
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["total_bytes"] == local.total_bytes


@pytest.mark.parametrize("backend", ["memfd", "disk"])
def test_copy_workers_hold_only_one_chunk_at_a_time(tmp_path, monkeypatch, backend):
    chunk = 1024 * 1024
    model = make_model_dir(tmp_path / "model", payload=b"x" * (3 * chunk))
    source = next(item for item in cache.collect_source_files(model) if item.rel.name == "model.safetensors")
    def fallback(*_):
        raise OSError("force read/write fallback")
    monkeypatch.setattr(cache, "_copy_with_copy_file_range", fallback)
    tracemalloc.start()
    try:
        if backend == "memfd":
            cached = cache.copy_file_to_memfd(source, chunk_size=chunk)
            cached.close()
        else:
            from infiniloader.local_cache import _copy_one
            _copy_one(source, tmp_path / "copy", chunk)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < chunk * 1.5, peak
    finally:
        tracemalloc.stop()
