import json
import os
from pathlib import Path
import time

import pytest

from conftest import make_model_dir
from infiniloader.cli import main
from infiniloader.local_cache import garbage_collect, lease_path, materialize_model_cache, verify_cache
from infiniloader.runtime import FileLock, atomic_json
from test_lifecycle import running_daemon

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX file locks")


def test_corruption_is_rejected_and_rebuilt_as_new_generation(tmp_path):
    model = make_model_dir(tmp_path / "model")
    first = materialize_model_cache(model, base_dir=tmp_path / "state")
    file = first.cache_path / "model.safetensors"
    before = file.stat()
    file.chmod(0o644)
    file.write_bytes(b"X" * before.st_size)
    os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not verify_cache(first.cache_path, full=False)["valid"]
    assert not verify_cache(first.cache_path)["valid"]
    second = materialize_model_cache(model, base_dir=tmp_path / "state")
    assert not second.reused
    assert second.cache_path != first.cache_path
    assert verify_cache(second.cache_path)["valid"]
    assert first.cache_path.exists()


def test_refresh_and_source_revision_preserve_old_reader_paths(tmp_path):
    model = make_model_dir(tmp_path / "model")
    first = materialize_model_cache(model, base_dir=tmp_path / "state")
    refreshed = materialize_model_cache(model, base_dir=tmp_path / "state", refresh=True)
    assert refreshed.cache_path != first.cache_path
    assert materialize_model_cache(model, base_dir=tmp_path / "state").cache_path == refreshed.cache_path
    (model / "model.safetensors").write_bytes(b"new model")
    changed = materialize_model_cache(model, base_dir=tmp_path / "state")
    assert changed.cache_path != refreshed.cache_path
    assert (first.cache_path / "model.safetensors").read_bytes() == b"0123456789abcdef"
    assert (changed.cache_path / "model.safetensors").read_bytes() == b"new model"


def test_gc_dry_run_lru_and_active_leases(tmp_path):
    root = tmp_path / "cache"
    first = materialize_model_cache(make_model_dir(tmp_path / "a"), cache_root=root)
    second = materialize_model_cache(make_model_dir(tmp_path / "b"), cache_root=root)
    atomic_json(root / ".usage" / f"{first.cache_path.name}.json", {"last_used_at": 1})
    preview = garbage_collect(cache_root=root, max_bytes=second.total_bytes)
    assert [x["cache_path"] for x in preview["candidates"]] == [str(first.cache_path)]
    assert first.cache_path.exists()
    with FileLock(lease_path(first.cache_path), shared=True):
        result = garbage_collect(cache_root=root, max_bytes=0, apply=True)
        assert not result["budget_met"]
        assert result["skipped"][0]["reason"] == "active lease"
        assert first.cache_path.exists()
        assert not second.cache_path.exists()
    assert garbage_collect(cache_root=root, ttl=0, apply=True)["remaining_bytes"] == 0


def test_daemon_pins_local_disk_generation(tmp_path):
    model = make_model_dir(tmp_path / "model")
    local = materialize_model_cache(model, base_dir=tmp_path / "state")
    with running_daemon(tmp_path, local.cache_path, extra=["--local-cache-path", str(local.cache_path),
                                                         "--upstream-source", str(model)]):
        assert not garbage_collect(base_dir=tmp_path / "state", max_bytes=0, apply=True)["budget_met"]
    assert main(["verify", str(local.cache_path), "--json"]) == 0


def test_verify_rejects_manifest_traversal(tmp_path):
    local = materialize_model_cache(make_model_dir(tmp_path / "model"), base_dir=tmp_path / "state")
    manifest_file = local.cache_path / ".iloader-cache/manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["cached_files"][0]["relative_path"] = "../outside"
    atomic_json(manifest_file, manifest)
    assert not verify_cache(local.cache_path)["valid"]


def test_full_verify_checks_content_even_if_metadata_matches_manifest(tmp_path):
    local = materialize_model_cache(make_model_dir(tmp_path / "model"), base_dir=tmp_path / "state")
    manifest_file = local.cache_path / ".iloader-cache/manifest.json"
    manifest = json.loads(manifest_file.read_text())
    item = manifest["cached_files"][0]
    file = local.cache_path / item["relative_path"]
    file.chmod(0o644)
    file.write_bytes(b"X" * item["size"])
    current = file.stat()
    item.update(mtime_ns=current.st_mtime_ns, ctime_ns=current.st_ctime_ns)
    atomic_json(manifest_file, manifest)
    assert verify_cache(local.cache_path, full=False)["valid"]
    assert any("SHA-256 mismatch" in error for error in verify_cache(local.cache_path)["errors"])
