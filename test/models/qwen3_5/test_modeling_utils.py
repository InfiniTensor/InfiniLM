import torch
from infinilm.modeling_utils import _remap_qwen3_5, load_state_dict
from safetensors.torch import save_file


def test_remap_materializes_only_unit_offset_rmsnorm_weights():
    final_norm = "model.language_model.norm.weight"
    input_norm = "model.language_model.layers.0.input_layernorm.weight"
    q_norm = "model.language_model.layers.3.self_attn.q_norm.weight"
    gated_norm = "model.language_model.layers.0.linear_attn.norm.weight"
    vision_norm = "model.visual.blocks.0.norm1.weight"
    state_dict = {
        final_norm: torch.tensor([0.25, -0.5]),
        input_norm: torch.tensor([0.0, 0.5]),
        q_norm: torch.tensor([-0.25, 0.75]),
        gated_norm: torch.tensor([0.8, 1.2]),
        vision_norm: torch.tensor([0.9, 1.1]),
    }
    config = {
        "text_config": {
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 4,
        }
    }

    remapped = _remap_qwen3_5(state_dict, config)

    torch.testing.assert_close(remapped[final_norm], torch.tensor([1.25, 0.5]))
    torch.testing.assert_close(remapped[input_norm], torch.tensor([1.0, 1.5]))
    torch.testing.assert_close(remapped[q_norm], torch.tensor([0.75, 1.75]))
    torch.testing.assert_close(remapped[gated_norm], torch.tensor([0.8, 1.2]))
    torch.testing.assert_close(remapped[vision_norm], torch.tensor([0.9, 1.1]))


def test_remap_supports_text_only_final_norm_name():
    state_dict = {"model.norm.weight": torch.tensor([0.0, -0.25])}
    config = {"text_config": {"linear_key_head_dim": 128, "linear_num_key_heads": 4}}

    remapped = _remap_qwen3_5(state_dict, config)
    torch.testing.assert_close(remapped["model.norm.weight"], torch.tensor([1.0, 0.75]))


def test_load_state_dict_preserves_qwen_gdn_fp32_only_when_requested(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    gdn_key = "model.layers.0.linear_attn.A_log"
    router_key = "model.layers.0.mlp.gate.e_score_correction_bias"
    regular_key = "model.layers.0.self_attn.q_proj.weight"
    save_file(
        {
            gdn_key: torch.ones(2, dtype=torch.float32),
            router_key: torch.ones(2, dtype=torch.float32),
            regular_key: torch.ones(2, dtype=torch.float32),
        },
        checkpoint,
    )

    default = load_state_dict(checkpoint, dtype=torch.bfloat16)
    ascend = load_state_dict(
        checkpoint, dtype=torch.bfloat16, preserve_qwen_gdn_fp32=True
    )

    assert default[gdn_key].dtype == torch.bfloat16
    assert default[router_key].dtype == torch.float32
    assert ascend[gdn_key].dtype == torch.float32
    assert ascend[router_key].dtype == torch.float32
    assert ascend[regular_key].dtype == torch.bfloat16


def test_remap_splits_fused_moe_expert_weights_into_loader_views():
    prefix = "model.language_model.layers.0.mlp.experts."
    gate_up = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    down = torch.arange(2 * 3 * 2).reshape(2, 3, 2)
    state_dict = {
        prefix + "gate_up_proj": gate_up,
        prefix + "down_proj": down,
    }
    config = {
        "text_config": {
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 4,
        }
    }

    remapped = _remap_qwen3_5(state_dict, config)

    expected_keys = {
        prefix + f"{expert_id}.{projection}.weight"
        for expert_id in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    assert set(remapped) == expected_keys
    torch.testing.assert_close(remapped[prefix + "0.gate_proj.weight"], gate_up[0, :2])
    torch.testing.assert_close(remapped[prefix + "1.up_proj.weight"], gate_up[1, 2:])
    torch.testing.assert_close(remapped[prefix + "1.down_proj.weight"], down[1])
