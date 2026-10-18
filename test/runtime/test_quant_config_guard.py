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
