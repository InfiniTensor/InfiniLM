import os
import tempfile
import unittest

import torch
from infinilm.modeling_utils import load_state_dict
from safetensors.torch import save_file


class LoadStateDictTest(unittest.TestCase):
    def test_converts_weight_scale_to_float32(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "model.safetensors")
            save_file(
                {
                    "model.layers.0.self_attn.q_proj.weight_scale": torch.ones(
                        4, 1, dtype=torch.bfloat16
                    ),
                    "model.layers.0.input_layernorm.weight": torch.ones(
                        4, dtype=torch.bfloat16
                    ),
                },
                checkpoint_path,
                metadata={"format": "pt"},
            )

            state_dict = load_state_dict(checkpoint_path, dtype=torch.float16)

        self.assertEqual(
            state_dict["model.layers.0.self_attn.q_proj.weight_scale"].dtype,
            torch.float32,
        )
        self.assertEqual(
            state_dict["model.layers.0.input_layernorm.weight"].dtype,
            torch.float16,
        )


if __name__ == "__main__":
    unittest.main()
