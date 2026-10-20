import pytest
import torch

import infini.ops

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_CASES = (
    ((4, 4), None, None, 0, False),
    ((4, 4), None, None, -1, True),
    ((3, 4, 5), (30, 6, 1), (30, 6, 1), 1, False),
    ((2, 3, 7), None, None, -2, False),
)

_FLOAT_DTYPE_CASES = (
    (torch.float32, 1e-5, 3e-5),
    (torch.float16, 1e-2, 1e-3),
    (torch.bfloat16, 1e-2, 1e-2),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape, input_strides, out_strides, dim, inplace", _CASES)
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _FLOAT_DTYPE_CASES)
def test_softmax(
    shape,
    input_strides,
    out_strides,
    dim,
    inplace,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    out = (
        input
        if inplace
        else empty_strided(shape, out_strides, dtype=dtype, device=device)
    )

    return Payload(
        _softmax,
        _torch_softmax,
        (input, dim, None, out),
        {"implementation_index": implementation_index},
        rtol=rtol,
        atol=atol,
    )


def _softmax(input, dim, dtype, out, *, implementation_index):
    infini.ops.softmax(
        input,
        dim,
        dtype,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_softmax(input, dim, dtype, out, *, implementation_index):
    del implementation_index

    out.copy_(torch.softmax(input, dim=dim, dtype=dtype))

    return out
