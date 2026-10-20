from __future__ import annotations

import os

import pytest

from conftest import make_model_dir, write_safetensors
from infiniloader.distributed import build_tp_plan, create_rank_views, simulate_distributed_read


@pytest.mark.skipif(os.name != "posix", reason="rank views use POSIX symlinks")
def test_create_rank_views_and_simulate_read(tmp_path):
    model = make_model_dir(tmp_path / "model", payload=b"x" * 4096)
    manifest = create_rank_views(model, name="tp2", ranks=2, base_dir=tmp_path / "state")
    rank_root = tmp_path / "state" / "dist-views" / "tp2"

    assert manifest["ranks"] == 2
    assert (rank_root / ".iloader-dist" / "manifest.json").exists()
    assert (rank_root / "rank0" / ".iloader-dist" / "rank.json").exists()
    assert (rank_root / "rank0" / "config.json").read_text(encoding="utf-8") == '{"model_type":"unit"}\n'

    result = simulate_distributed_read(
        model,
        ranks=2,
        rank_view_root=rank_root,
        include_patterns=("*.safetensors",),
        chunk_size=1024,
    )

    assert result["ok"] is True
    assert result["aggregate_bytes"] == 4096 * 2


def test_build_tp_plan_from_safetensors_header(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    write_safetensors(
        model / "model.safetensors",
        {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32",
                "shape": [8, 4],
                "data_offsets": [0, 128],
            },
            "model.layers.0.input_layernorm.weight": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [128, 144],
            },
        },
        payload_size=144,
    )

    plan = build_tp_plan(model, tp=2)
    by_name = {item["name"]: item for item in plan["tensors"]}

    q_proj = by_name["model.layers.0.self_attn.q_proj.weight"]
    norm = by_name["model.layers.0.input_layernorm.weight"]
    assert plan["tensor_count"] == 2
    assert q_proj["action"] == "split"
    assert q_proj["dim"] == 0
    assert q_proj["per_rank_shape"] == [4, 4]
    assert norm["action"] == "replicate"
