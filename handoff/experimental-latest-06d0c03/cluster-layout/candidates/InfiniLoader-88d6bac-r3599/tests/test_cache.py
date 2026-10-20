from __future__ import annotations

import mmap
import os

import pytest

from conftest import make_model_dir
from infiniloader.cache import collect_source_files, load_model_into_memory, build_view
from infiniloader.local_cache import cache_status, list_model_caches, materialize_model_cache


def test_collect_source_files_applies_default_excludes(tmp_path):
    model = make_model_dir(tmp_path / "model")

    rels = {item.rel.as_posix() for item in collect_source_files(model)}

    assert "config.json" in rels
    assert "tokenizer_config.json" in rels
    assert "model.safetensors" in rels
    assert "nested/extra.bin" in rels
    assert ".iloader-cache/ignored.bin" not in rels
    assert "debug.log" not in rels


def test_materialize_model_cache_reuses_valid_cache(tmp_path):
    model = make_model_dir(tmp_path / "model")
    base_dir = tmp_path / "state"

    first = materialize_model_cache(model, name="unit", base_dir=base_dir, workers=1, chunk_size=4)
    second = materialize_model_cache(model, name="unit", base_dir=base_dir, workers=1, chunk_size=4)
    status = cache_status(model, name="unit", base_dir=base_dir)
    caches = list_model_caches(base_dir=base_dir)

    assert first.reused is False
    assert second.reused is True
    assert first.cache_path == second.cache_path
    assert (first.cache_path / "config.json").read_text(encoding="utf-8") == '{"model_type":"unit"}\n'
    assert (first.cache_path / ".iloader-cache" / "manifest.json").exists()
    assert status["valid"] is True
    assert len(caches) == 1


@pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="memfd_create is Linux-only")
def test_memfd_view_round_trip(tmp_path):
    payload = b"abcd" * 4096
    model = make_model_dir(tmp_path / "model", payload=payload)
    view = tmp_path / "view"

    cached = load_model_into_memory(model, workers=2, chunk_size=1024)
    try:
        build_view(cached, view)

        assert (view / ".iloader" / "manifest.json").exists()
        assert (view / "model.safetensors").read_bytes() == payload
        with (view / "model.safetensors").open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                assert mm[:8] == b"abcdabcd"
                assert len(mm) == len(payload)
            finally:
                mm.close()
    finally:
        cached.close()
