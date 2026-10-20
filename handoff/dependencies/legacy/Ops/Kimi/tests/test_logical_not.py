import infini.ops
import pytest
import torch


@pytest.mark.parametrize("input_dtype", (torch.bool, torch.uint8))
@pytest.mark.parametrize("strided", (False, True))
def test_bool_output_distinct_from_uint8(input_dtype, strided, device):
    values = torch.tensor([[0, 1, 2], [1, 0, 0]], dtype=input_dtype, device=device)
    input = values.t() if strided else values
    expected = torch.logical_not(input)
    out = torch.empty_like(expected)

    infini.ops.logical_not(input, out, implementation_index=8)

    assert out.dtype == torch.bool
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
