import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import gguf_mapping as mapping  # noqa: E402
import gguf_transforms as transforms  # noqa: E402


class GGUFTransformsTest(unittest.TestCase):
    def test_value_head_permutation_round_trip(self):
        source = np.arange(2 * 3 * 4 * 5, dtype=np.uint8).reshape(24, 5)
        tiled = transforms.reorder_v(source, n_k=2, n_v_per_k=3, hd=4)
        restored = transforms.reorder_v_inverse(tiled, n_k=2, n_v_per_k=3, hd=4)

        np.testing.assert_array_equal(restored, source)
        self.assertEqual(sorted(map(bytes, tiled)), sorted(map(bytes, source)))

    def test_tail_permutation_preserves_prefix_and_rows(self):
        dims = SimpleNamespace(lin_k_heads=2, lin_v_heads=6, value_dim=12)
        entry = SimpleNamespace(shape=(20, 3), vperm="v_tail", infinilm="conv")
        source = np.arange(60, dtype=np.uint8).reshape(20, 3)

        tiled = transforms.apply_vperm(source, entry, dims, direction="fwd")
        restored = transforms.apply_vperm(tiled, entry, dims, direction="inv")

        np.testing.assert_array_equal(tiled[:8], source[:8])
        np.testing.assert_array_equal(restored, source)
        self.assertEqual(sorted(map(bytes, tiled[8:])), sorted(map(bytes, source[8:])))

    def test_bf16_bits_for_exact_values(self):
        values = np.array([0.0, 1.0, -2.0, np.inf], dtype=np.float32)
        expected = np.array([0x0000, 0x3F80, 0xC000, 0x7F80], dtype=np.uint16)
        np.testing.assert_array_equal(transforms.bf16_bits(values), expected)


class GGUFMappingTest(unittest.TestCase):
    def test_generated_config_keeps_quantization_at_root(self):
        table = {
            "model.language_model.layers.0.mlp.down_proj.weight_bytes": mapping.Q6_K
        }
        rules = mapping.activation_vperm_rules(
            mapping.REAL, mapping.build_plan(mapping.REAL)
        )
        config = mapping.make_root_config(mapping.REAL, table, rules)

        self.assertNotIn("quantization_config", config["text_config"])
        quant = config["quantization_config"]
        self.assertEqual(quant["quant_method"], "gguf")
        self.assertEqual(quant["key_prefix"], mapping.PREFIX)
        self.assertEqual(quant["ggml_types"], table)
        self.assertEqual(quant["activation_vperm"], rules)
        self.assertTrue(rules)
        self.assertEqual(len({rule["suffix"] for rule in rules}), len(rules))

    def test_packed_checkpoint_name_and_row_size(self):
        entry = SimpleNamespace(
            blob=True, infinilm="model.language_model.layers.0.mlp.down_proj.weight"
        )
        self.assertEqual(
            mapping.ckpt_name(entry),
            "model.language_model.layers.0.mlp.down_proj.weight_bytes",
        )
        self.assertEqual(mapping.row_bytes(5120, block_size=256, type_size=210), 4200)
        with self.assertRaises(ValueError):
            mapping.row_bytes(5119, block_size=256, type_size=210)


if __name__ == "__main__":
    unittest.main()
