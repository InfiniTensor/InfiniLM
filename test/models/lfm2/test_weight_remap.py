"""Unit tests for the LFM2 checkpoint-name adapter.

This test deliberately stubs the compiled ``infinicore`` module so the name
mapping can be checked before a native InfiniLM build is available.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def _load_modeling_utils():
    fake_infinicore = types.ModuleType("infinicore")
    fake_infinicore.float32 = object()
    fake_infinicore.float16 = object()
    fake_infinicore.bfloat16 = object()
    fake_infinicore.int8 = object()
    fake_infinicore.int32 = object()
    fake_infinicore.int64 = object()
    fake_infinicore.dtype = object()
    fake_infinicore.device = object()
    fake_infinicore.Tensor = object
    fake_infinicore.nn = types.SimpleNamespace(Module=object)
    sys.modules.setdefault("infinicore", fake_infinicore)

    module_path = (
        Path(__file__).resolve().parents[3]
        / "python"
        / "infinilm"
        / "modeling_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "infinilm_lfm2_modeling_utils", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Lfm2WeightRemapTest(unittest.TestCase):
    def test_all_released_name_patterns(self):
        module = _load_modeling_utils()
        tensor = torch.zeros(1)
        state_dict = {
            "model.embedding_norm.weight": tensor,
            "model.layers.2.self_attn.q_layernorm.weight": tensor,
            "model.layers.2.self_attn.k_layernorm.weight": tensor,
            "model.layers.2.self_attn.out_proj.weight": tensor,
            "model.layers.0.feed_forward.w1.weight": tensor,
            "model.layers.0.feed_forward.w2.weight": tensor,
            "model.layers.0.feed_forward.w3.weight": tensor,
            "model.layers.0.conv.conv.weight": tensor,
            "model.layers.0.conv.in_proj.weight": tensor,
            "model.layers.0.conv.out_proj.weight": tensor,
            "model.layers.0.operator_norm.weight": tensor,
            "model.layers.0.ffn_norm.weight": tensor,
            "model.embed_tokens.weight": tensor,
        }

        remapped = module._remap_lfm2(state_dict)
        self.assertEqual(
            set(remapped),
            {
                "model.norm.weight",
                "model.layers.2.self_attn.q_norm.weight",
                "model.layers.2.self_attn.k_norm.weight",
                "model.layers.2.self_attn.o_proj.weight",
                "model.layers.0.feed_forward.gate_proj.weight",
                "model.layers.0.feed_forward.down_proj.weight",
                "model.layers.0.feed_forward.up_proj.weight",
                "model.layers.0.conv.conv.weight",
                "model.layers.0.conv.in_proj.weight",
                "model.layers.0.conv.out_proj.weight",
                "model.layers.0.operator_norm.weight",
                "model.layers.0.ffn_norm.weight",
                "model.embed_tokens.weight",
            },
        )

    def test_registry_selects_lfm2_remapper(self):
        module = _load_modeling_utils()
        self.assertIs(module._WEIGHT_REMAPPER["lfm2"], module._remap_lfm2)


if __name__ == "__main__":
    unittest.main()
