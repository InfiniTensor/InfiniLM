"""Model construction must reject unimplemented or malformed quantization configs."""

import json
import unittest

from infinilm._core import device
from infinilm.lib import _infinilm


class QuantConfigGuardTest(unittest.TestCase):
    def test_quantized_checkpoint_cannot_silently_use_dense_weights(self):
        cases = (
            (
                {"quant_method": "unknown-format"},
                "Unsupported quantization method: unknown-format",
            ),
            ({}, "nonempty string quant_method"),
            ({"quant_method": 7}, "nonempty string quant_method"),
            ([], "nonempty string quant_method"),
        )
        for quantization_config, message in cases:
            with self.subTest(quantization_config=quantization_config):
                config = {
                    "model_type": "qwen3",
                    "vocab_size": 32,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "max_position_embeddings": 32,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                    "hidden_act": "silu",
                    "torch_dtype": "float32",
                    "quantization_config": quantization_config,
                }
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    _infinilm.InferEngine(
                        json.dumps(config),
                        _infinilm.DistConfig(1),
                        device("cpu")._underlying.type,
                    )


if __name__ == "__main__":
    unittest.main()
