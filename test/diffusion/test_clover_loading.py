"""Protect the text-only Clover checkpoint layout and learned preview weights."""

import numpy as np
from infinilm import _core as core
from infinilm.modeling_utils import _remap_clover_moe


def test_clover_remap_preserves_preview_weights_and_splits_qkv():
    config = {
        "linear_key_head_dim": 2,
        "linear_num_key_heads": 1,
        "num_experts": 2,
        "moe_intermediate_size": 2,
    }
    prefix = "model.layers.0.linear_attn."
    norm = core.from_list([0.25, -0.5], dtype=core.float32)
    scalar = core.from_list([5.0, 6.0], dtype=core.float32)
    packed = core.from_numpy(np.arange(24, dtype=np.float32).reshape(6, 4))
    output = _remap_clover_moe(
        {
            "model.norm.weight": norm,
            prefix + "norm.weight": norm,
            prefix + "bd_softmax_history_bias": scalar,
            prefix + "in_proj_qkv.weight": packed,
        },
        config,
    )
    np.testing.assert_array_equal(output["model.norm.weight"].to_numpy(), [1.25, 0.5])
    np.testing.assert_array_equal(
        output[prefix + "norm.weight"].to_numpy(), [0.25, -0.5]
    )
    assert output[prefix + "bd_softmax_history_bias"] is scalar
    assert prefix + "in_proj_qkv.weight" not in output
    for part, begin in (("q", 0), ("k", 2), ("v", 4)):
        np.testing.assert_array_equal(
            output[prefix + f"in_proj_{part}.weight"].to_numpy(),
            packed.to_numpy()[begin : begin + 2],
        )
