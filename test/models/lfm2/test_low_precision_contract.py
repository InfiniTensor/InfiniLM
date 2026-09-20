"""Reference rounding boundaries, independent of GPU and native extensions."""

import unittest

import torch
import torch.nn.functional as F


class Lfm2LowPrecisionContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260915)

    def test_rms_norm_casts_before_learned_weight(self):
        value = torch.randn(2, 5, 16).bfloat16()
        weight = torch.randn(16).bfloat16()
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + 1e-5
        )
        expected = normalized.bfloat16() * weight
        unit_norm = (normalized * torch.ones_like(weight).float()).bfloat16()
        torch.testing.assert_close(unit_norm * weight, expected, rtol=0, atol=0)

    def test_mlp_materializes_silu_before_product(self):
        gate, up = torch.randn(2, 5, 16).bfloat16(), torch.randn(2, 5, 16).bfloat16()
        actual = F.silu(gate.float()).bfloat16() * up
        expected = F.silu(gate) * up
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_neox_rotary_rounds_tables_and_both_products(self):
        value = torch.randn(2, 5, 16).bfloat16()
        cosine, sine = torch.randn(5, 8).bfloat16(), torch.randn(5, 8).bfloat16()
        first, second = value.chunk(2, -1)
        actual = torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine), -1
        )
        rotated_half = torch.cat((-second, first), -1)
        expected = value * torch.cat((cosine, cosine), -1) + rotated_half * torch.cat(
            (sine, sine), -1
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_python_scalar_keeps_float32_opmath(self):
        value = torch.arange(1, 1000).bfloat16()
        alpha = 8**-0.5
        expected = (value.float() * alpha).bfloat16()
        torch.testing.assert_close(value * alpha, expected, rtol=0, atol=0)
        rounded_alpha = value * torch.tensor(alpha).bfloat16()
        self.assertGreater(int(torch.count_nonzero(rounded_alpha != expected)), 0)

    def test_attention_rounds_qk_before_scale(self):
        query, key = torch.randn(2, 5, 8).bfloat16(), torch.randn(2, 8, 5).bfloat16()
        alpha = 8**-0.5
        expected = (query.float() @ key.float()).bfloat16() * alpha
        actual = (query @ key) * alpha
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        fused_scale = ((query.float() @ key.float()) * alpha).bfloat16()
        self.assertGreater(int(torch.count_nonzero(fused_scale != expected)), 0)


if __name__ == "__main__":
    unittest.main()
