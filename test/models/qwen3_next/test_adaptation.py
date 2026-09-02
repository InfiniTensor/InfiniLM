import unittest

import torch
from infinilm.modeling_utils import _remap_qwen3_next


class Qwen3NextWeightRemapTest(unittest.TestCase):
    def setUp(self):
        self.num_key_heads = 2
        self.num_value_heads = 4
        self.key_head_dim = 2
        self.value_head_dim = 3
        self.config = {
            "linear_num_key_heads": self.num_key_heads,
            "linear_num_value_heads": self.num_value_heads,
            "linear_key_head_dim": self.key_head_dim,
            "linear_value_head_dim": self.value_head_dim,
        }
        self.prefix = "model.layers.0.linear_attn."
        self.hidden_size = 3

    def test_deinterleaves_qkvz_by_key_head_group(self):
        values_per_key = self.num_value_heads // self.num_key_heads
        value_group_dim = values_per_key * self.value_head_dim
        group_size = 2 * self.key_head_dim + 2 * value_group_dim
        packed = torch.arange(
            self.num_key_heads * group_size * self.hidden_size,
        ).reshape(self.num_key_heads * group_size, self.hidden_size)
        grouped = packed.view(self.num_key_heads, group_size, self.hidden_size)

        remapped = _remap_qwen3_next(
            {f"{self.prefix}in_proj_qkvz.weight": packed},
            self.config,
        )

        q_end = self.key_head_dim
        k_end = q_end + self.key_head_dim
        v_end = k_end + value_group_dim
        expected_sizes = {
            "q": (0, q_end),
            "k": (q_end, k_end),
            "v": (k_end, v_end),
            "z": (v_end, group_size),
        }
        for name, (start, end) in expected_sizes.items():
            expected = grouped[:, start:end].reshape(-1, self.hidden_size)
            self.assertTrue(
                torch.equal(remapped[f"{self.prefix}in_proj_{name}.weight"], expected)
            )
        self.assertNotIn(f"{self.prefix}in_proj_qkvz.weight", remapped)

    def test_deinterleaves_ba_by_key_head_group(self):
        values_per_key = self.num_value_heads // self.num_key_heads
        group_size = 2 * values_per_key
        packed = torch.arange(
            self.num_key_heads * group_size * self.hidden_size,
        ).reshape(self.num_key_heads * group_size, self.hidden_size)
        grouped = packed.view(self.num_key_heads, group_size, self.hidden_size)

        remapped = _remap_qwen3_next(
            {f"{self.prefix}in_proj_ba.weight": packed},
            self.config,
        )

        expected_b = grouped[:, :values_per_key].reshape(-1, self.hidden_size)
        expected_a = grouped[:, values_per_key:].reshape(-1, self.hidden_size)
        self.assertTrue(
            torch.equal(remapped[f"{self.prefix}in_proj_b.weight"], expected_b)
        )
        self.assertTrue(
            torch.equal(remapped[f"{self.prefix}in_proj_a.weight"], expected_a)
        )
        self.assertNotIn(f"{self.prefix}in_proj_ba.weight", remapped)


if __name__ == "__main__":
    unittest.main()
