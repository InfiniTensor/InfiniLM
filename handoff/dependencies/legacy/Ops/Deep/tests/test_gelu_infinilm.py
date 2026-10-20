import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides, inplace",
    (
        ((13, 4), None, None, False),
        ((13, 4), None, None, True),
        ((13, 4), (10, 1), (10, 1), False),
        ((13, 4), (10, 1), (10, 1), True),
        ((13, 4, 4), None, None, False),
        ((13, 4, 4), None, None, True),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), False),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), True),
        ((16, 5632), None, None, False),
        ((16, 5632), None, None, True),
        ((16, 5632), (13312, 1), (13312, 1), False),
        ((16, 5632), (13312, 1), (13312, 1), True),
        ((4, 4, 5632), None, None, False),
        ((4, 4, 5632), None, None, True),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), False),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), True),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float64, 1e-6, 1e-6),
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_gelu_infinilm(
    shape, input_strides, out_strides, inplace, dtype, device, rtol, atol
):
    if device == "musa" and dtype == torch.float64:
        pytest.skip("MUSA does not support float64 GELU_INFINILM")

    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    out = (
        input
        if inplace
        else empty_strided(shape, out_strides, dtype=dtype, device=device)
    )

    return Payload(
        _gelu_infinilm,
        _torch_gelu_infinilm,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _gelu_infinilm(input, out):
    infini.ops.gelu_infinilm(input, "none", out, stream=get_stream(input.device))

    return out


def _torch_gelu_infinilm(input, out):
    result = torch.nn.functional.gelu(input, approximate="none")
    out.copy_(result)

    return out
