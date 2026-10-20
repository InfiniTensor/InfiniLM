import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    "use_deprecated, decimals",
    (
        (False, 0),
        (True, 2),
    ),
)
def test_special_round(
    use_deprecated,
    decimals,
    dtype,
    rtol,
    atol,
    device,
    implementation_index,
):
    input = torch.tensor((1.234, -2.675, 3.5), dtype=dtype, device=device)
    out = torch.empty_like(input)
    kwargs = {
        "stream": get_stream(device),
        "implementation_index": implementation_index,
    }

    if use_deprecated:
        infini.ops.special_round(input, decimals, out, **kwargs)
        expected = torch.round(input, decimals=decimals)
    else:
        infini.ops.special_round(input, out, **kwargs)
        expected = torch.special.round(input)

    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
