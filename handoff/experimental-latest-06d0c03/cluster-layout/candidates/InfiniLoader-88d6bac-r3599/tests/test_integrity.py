import errno
import os

import pytest

from conftest import make_model_dir
from infiniloader.cache import build_view, collect_source_files, copy_file_to_memfd, load_model_into_memory

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux memfd tests")


def test_published_weights_cannot_be_modified_or_truncated(tmp_path):
    import fcntl

    model = make_model_dir(tmp_path / "model")
    cached = load_model_into_memory(model)
    try:
        view = build_view(cached, tmp_path / "view")
        target = view / "model.safetensors"
        original = target.read_bytes()
        item = next(f for f in cached.files if f.source.rel.name == target.name)
        assert fcntl.fcntl(item.fd, fcntl.F_GET_SEALS) & fcntl.F_SEAL_WRITE
        with pytest.raises(OSError):
            with target.open("r+b") as stream:
                stream.write(b"bad")
                stream.flush()
        with pytest.raises(OSError):
            os.ftruncate(item.fd, 0)
        assert target.read_bytes() == original
    finally:
        cached.close()


def test_missing_source_does_not_leak_memfd(tmp_path):
    model = make_model_dir(tmp_path / "model")
    source = collect_source_files(model)[0]
    source.path.unlink()
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(FileNotFoundError):
        copy_file_to_memfd(source)
    assert len(os.listdir("/proc/self/fd")) == before


def test_source_mutation_aborts_preload(tmp_path, monkeypatch):
    from infiniloader import cache

    model = make_model_dir(tmp_path / "model")
    original = cache._copy_with_read_write
    def copy_and_mutate(src, dst, size, chunk):
        result = original(src, dst, size, chunk)
        (model / "model.safetensors").write_bytes(b"different-source")
        return result
    monkeypatch.setattr(cache, "_copy_with_copy_file_range", lambda *a: (_ for _ in ()).throw(OSError(errno.EXDEV, "fallback")))
    monkeypatch.setattr(cache, "_copy_with_read_write", copy_and_mutate)
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(RuntimeError, match="changed"):
        load_model_into_memory(model)
    assert len(os.listdir("/proc/self/fd")) == before


def test_source_replacement_with_same_size_and_mtime_is_detected(tmp_path):
    model = make_model_dir(tmp_path / "model")
    source = collect_source_files(model)[0]
    source.path.unlink()
    source.path.write_bytes(b"x" * source.size)
    os.utime(source.path, ns=(source.mtime_ns, source.mtime_ns))
    with pytest.raises(RuntimeError, match="replaced"):
        copy_file_to_memfd(source)
