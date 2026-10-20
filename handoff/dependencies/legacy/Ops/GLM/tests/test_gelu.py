import pytest
import torch

import infini.ops

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_SHAPE_CASES = (
    ((), None, None, False),
    ((0,), None, None, False),
    ((13, 4), None, None, False),
    ((13, 4), None, None, True),
    ((13, 4), (10, 1), (10, 1), False),
    ((13, 4), (10, 1), (10, 1), True),
    ((4, 4, 5632), None, None, False),
)

_FLOAT_DTYPE_CASES = (
    (torch.float64, 1e-6, 1e-6),
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
    (torch.bfloat16, 1e-2, 1e-2),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape, input_strides, out_strides, inplace", _SHAPE_CASES)
@pytest.mark.parametrize("approximate", ("none", "tanh"))
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _FLOAT_DTYPE_CASES)
def test_gelu(
    shape,
    input_strides,
    out_strides,
    inplace,
    approximate,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    if device == "musa" and dtype == torch.float64:
        pytest.skip("MUSA does not support float64 GELU")
    if device == "mlu" and dtype == torch.float64:
        pytest.skip("Cambricon CNNL does not support float64 GELU")

    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    out = (
        input
        if inplace
        else empty_strided(shape, out_strides, dtype=dtype, device=device)
    )

    return Payload(
        lambda *args: _gelu(*args, implementation_index=implementation_index),
        _torch_gelu,
        (input, approximate, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _gelu(input, approximate, out, *, implementation_index=0):
    infini.ops.gelu(
        input,
        approximate,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_gelu(input, approximate, out):
    out.copy_(torch.nn.functional.gelu(input, approximate=approximate))

    return out
