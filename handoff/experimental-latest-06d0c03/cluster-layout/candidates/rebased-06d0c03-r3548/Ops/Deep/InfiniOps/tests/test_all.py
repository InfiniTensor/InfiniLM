import infini.ops
import pytest
import torch


@pytest.mark.parametrize("input_dtype", (torch.bool, torch.uint8))
def test_bool_and_uint8_reductions_remain_distinct(input_dtype, device):
    input = torch.tensor([1, 0, 2], dtype=input_dtype, device=device)
    expected = torch.all(input, dim=0, keepdim=False)
    out = torch.empty_like(expected)

    infini.ops.all(input, 0, False, out, implementation_index=8)

    assert out.dtype == expected.dtype
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
