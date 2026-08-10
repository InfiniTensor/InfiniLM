import torch
from infinilm.modeling_utils import _remap_qwen3_5


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
