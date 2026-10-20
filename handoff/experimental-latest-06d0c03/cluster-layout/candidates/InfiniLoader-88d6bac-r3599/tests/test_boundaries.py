import os

import pytest

from conftest import make_model_dir
from infiniloader.cache import build_view, load_model_into_memory
from infiniloader.distributed import create_rank_views

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux view tests")


def test_view_must_not_overlap_source(tmp_path):
    source = make_model_dir(tmp_path / "model")
    cached = load_model_into_memory(source)
    try:
        for view in (source, source / "view", source.parent):
            with pytest.raises(RuntimeError, match="overlap"):
                build_view(cached, view, replace=True)
        assert (source / "model.safetensors").exists()
    finally:
        cached.close()


def test_tmp_name_does_not_authorize_deleting_ordinary_dist_directory(tmp_path):
    source = make_model_dir(tmp_path / "model")
    base = tmp_path / "state"
    ordinary = base / "dist-views" / "normal.tmp.name"
    ordinary.mkdir(parents=True)
    (ordinary / "important").write_text("keep")
    with pytest.raises(RuntimeError, match="non-InfiniLoader"):
        create_rank_views(source, name="normal.tmp.name", ranks=2, base_dir=base, replace=True)
    assert (ordinary / "important").read_text() == "keep"
