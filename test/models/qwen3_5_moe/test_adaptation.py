import unittest

import torch
from infinilm.infer_engine import _infer_position_id_axes
from infinilm.modeling_utils import _remap_qwen3_5_moe


class PositionIdAxesTest(unittest.TestCase):
    def test_defaults_to_one_axis(self):
        self.assertEqual(_infer_position_id_axes({"text_config": {}}), 1)

    def test_infers_axes_from_mrope_section(self):
        config = {"text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}}}
        self.assertEqual(_infer_position_id_axes(config), 3)

    def test_explicit_axes_take_precedence(self):
        config = {
            "position_id_axes": 2,
            "text_config": {
                "position_id_axes": 4,
                "rope_parameters": {"mrope_section": [11, 11, 10]},
            },
        }
        self.assertEqual(_infer_position_id_axes(config), 4)

    def test_rejects_non_positive_axes(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _infer_position_id_axes({"text_config": {"position_id_axes": 0}})


class Qwen35MoeWeightRemapTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "text_config": {
                "linear_key_head_dim": 2,
                "linear_num_key_heads": 1,
                "num_experts": 2,
                "moe_intermediate_size": 3,
            }
        }

    def test_splits_packed_expert_weights(self):
        gate_up = torch.arange(2 * 6 * 4).reshape(2, 6, 4)
        down = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": down,
        }

        remapped = _remap_qwen3_5_moe(state_dict, self.config)

        prefix = "model.language_model.layers.0.mlp.experts."
        self.assertTrue(
            torch.equal(remapped[f"{prefix}0.gate_proj.weight"], gate_up[0, :3])
        )
        self.assertTrue(
            torch.equal(remapped[f"{prefix}0.up_proj.weight"], gate_up[0, 3:])
        )
        self.assertTrue(torch.equal(remapped[f"{prefix}1.down_proj.weight"], down[1]))
        self.assertNotIn(f"{prefix}gate_up_proj", remapped)
        self.assertNotIn(f"{prefix}down_proj", remapped)

    def test_rejects_wrong_expert_count(self):
        state_dict = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.zeros(
                1, 6, 4
            )
        }
        with self.assertRaisesRegex(ValueError, "Expected 2 experts"):
            _remap_qwen3_5_moe(state_dict, self.config)


if __name__ == "__main__":
    unittest.main()
