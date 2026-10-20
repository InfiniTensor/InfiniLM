import infini.ops
import pytest
import torch


@pytest.mark.parametrize("input_dtype", (torch.uint8, torch.float32))
def test_bool_output(input_dtype, device):
    elements = torch.tensor([0, 1, 2, 1], dtype=input_dtype, device=device)
    test_elements = torch.tensor([1, 2], dtype=input_dtype, device=device)
    expected = torch.isin(elements, test_elements, assume_unique=False, invert=False)
    out = torch.empty_like(expected)

    infini.ops.isin(elements, test_elements, False, False, out, implementation_index=8)

    assert out.dtype == torch.bool
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
