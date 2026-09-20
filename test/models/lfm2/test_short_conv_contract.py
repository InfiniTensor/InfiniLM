"""Mathematical contract for the explicit LFM2 ShortConv decomposition."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F


def stateful_depthwise_conv(
    values: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the C++ implementation using [B, H, K-1] recurrent state."""
    kernel_size = weight.shape[-1]
    history = state.transpose(1, 2)
    combined = torch.cat((history, values), dim=1)

    if values.dtype != torch.float32 and values.shape[1] > 1:
        batch, sequence, hidden = values.shape
        windows = (
            combined.as_strided(
                (batch, hidden, sequence, kernel_size),
                ((sequence + kernel_size - 1) * hidden, 1, hidden, hidden),
            )
            .contiguous()
            .view(batch * hidden, sequence, kernel_size)
        )
        kernels = (
            weight.view(hidden, kernel_size, 1)
            .unsqueeze(0)
            .expand(batch, hidden, kernel_size, 1)
            .contiguous()
            .view(batch * hidden, kernel_size, 1)
        )
        output = torch.bmm(windows.float(), kernels.float()).to(values.dtype)
        output = output.view(batch, hidden, sequence).transpose(1, 2).contiguous()
        return output, combined[:, -kernel_size + 1 :, :].transpose(1, 2)

    output = None
    terms = []
    for kernel_idx in range(kernel_size):
        window = combined[:, kernel_idx : kernel_idx + values.shape[1], :]
        term = window * weight[:, 0, kernel_idx].view(1, 1, -1)
        terms.append(term)
        output = term if output is None else output + term

    if values.dtype != torch.float32:
        output = torch.stack(terms).float().sum(dim=0).to(values.dtype)

    newest_state = combined[:, -kernel_size + 1 :, :].transpose(1, 2)
    return output, newest_state


class Lfm2ShortConvContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260914)
        self.batch = 2
        self.sequence = 7
        self.hidden = 5
        self.kernel = 3
        self.values = torch.randn(self.batch, self.sequence, self.hidden)
        self.weight = torch.randn(self.hidden, 1, self.kernel)

    def test_prefill_matches_grouped_conv1d(self):
        state = torch.zeros(self.batch, self.hidden, self.kernel - 1)
        actual, _ = stateful_depthwise_conv(self.values, self.weight, state)
        expected = F.conv1d(
            self.values.transpose(1, 2),
            self.weight,
            padding=self.kernel - 1,
            groups=self.hidden,
        )[..., : self.sequence].transpose(1, 2)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_prefill_then_decode_matches_full_sequence(self):
        split = self.sequence - 1
        state = torch.zeros(self.batch, self.hidden, self.kernel - 1)
        prefill, state = stateful_depthwise_conv(
            self.values[:, :split], self.weight, state
        )
        decode, state = stateful_depthwise_conv(
            self.values[:, split:], self.weight, state
        )
        cached = torch.cat((prefill, decode), dim=1)

        full, expected_state = stateful_depthwise_conv(
            self.values,
            self.weight,
            torch.zeros_like(state),
        )
        torch.testing.assert_close(cached, full, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)

    def test_bf16_prefill_has_single_output_rounding(self):
        values, weight = self.values.bfloat16(), self.weight.bfloat16()
        state = torch.zeros(
            self.batch, self.hidden, self.kernel - 1, dtype=torch.bfloat16
        )
        actual, _ = stateful_depthwise_conv(values, weight, state)
        expected = F.conv1d(
            values.float().transpose(1, 2),
            weight.float(),
            padding=self.kernel - 1,
            groups=self.hidden,
        )[..., : self.sequence]
        torch.testing.assert_close(
            actual, expected.transpose(1, 2).bfloat16(), rtol=0, atol=0
        )

    def test_bf16_decode_sums_rounded_products_in_float32(self):
        values, weight = self.values[:, -1:].bfloat16(), self.weight.bfloat16()
        state = torch.randn(self.batch, self.hidden, self.kernel - 1).bfloat16()
        actual, newest = stateful_depthwise_conv(values, weight, state)
        combined = torch.cat((state, values.transpose(1, 2)), dim=-1)
        products = combined * weight[:, 0, :]
        expected = products.float().sum(-1).bfloat16().unsqueeze(1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(newest, combined[..., 1:], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
