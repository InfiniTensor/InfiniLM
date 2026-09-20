import json
import unittest

import torch

from infinilm.infer_engine import model_uses_mamba_cache
from infinilm.modeling_utils import _remap_mamba2
from scripts.prepare_mamba2_checkpoint import prepare_config


class Mamba2CheckpointTest(unittest.TestCase):
    def test_prepares_official_state_spaces_config(self):
        config = prepare_config(
            {
                "d_model": 768,
                "d_intermediate": 0,
                "n_layer": 24,
                "vocab_size": 50277,
                "pad_vocab_size_multiple": 16,
                "ssm_cfg": {"layer": "Mamba2"},
            }
        )

        self.assertEqual(config["model_type"], "mamba2")
        self.assertEqual(config["vocab_size"], 50288)
        self.assertEqual(config["hidden_size"], 768)
        self.assertEqual(config["intermediate_size"], 1536)
        self.assertEqual(config["state_size"], 128)
        self.assertEqual(config["num_heads"], 24)
        self.assertEqual(config["head_dim"], 64)

    def test_remaps_official_prefixes_and_expands_head_parameters(self):
        state_dict = {
            "backbone.embedding.weight": torch.zeros(32, 8),
            "backbone.layers.0.mixer.A_log": torch.tensor([1.0, 2.0]),
            "backbone.layers.0.mixer.D": torch.tensor([3.0, 4.0]),
            "backbone.layers.0.mixer.dt_bias": torch.tensor([5.0, 6.0]),
        }
        remapped = _remap_mamba2(
            state_dict,
            {"intermediate_size": 8, "num_heads": 2, "head_dim": 4},
        )

        self.assertIn("model.embedding.weight", remapped)
        self.assertIn("lm_head.weight", remapped)
        self.assertEqual(remapped["model.layers.0.mixer.A_log"].shape, (8,))
        torch.testing.assert_close(
            remapped["model.layers.0.mixer.A_log"],
            torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
        )

    def test_rejects_inconsistent_parameter_size(self):
        with self.assertRaisesRegex(ValueError, "expected 2 or 8"):
            _remap_mamba2(
                {"backbone.layers.0.mixer.A_log": torch.zeros(3)},
                {"intermediate_size": 8, "num_heads": 2, "head_dim": 4},
            )

    def test_model_uses_recurrent_state_cache(self):
        self.assertTrue(model_uses_mamba_cache({"model_type": "mamba2"}))


if __name__ == "__main__":
    unittest.main()
