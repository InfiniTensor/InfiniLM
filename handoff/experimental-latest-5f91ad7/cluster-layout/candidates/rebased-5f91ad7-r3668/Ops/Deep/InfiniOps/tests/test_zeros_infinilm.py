import infini.ops
import pytest
import torch

from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    randint_strided,
    randn_strided,
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides, inplace",
    (
        ((13, 4), None, None, False),
        ((13, 4), None, None, True),
        ((13, 4), (10, 1), (10, 1), False),
        ((13, 4), (0, 1), None, False),
        ((13, 4, 4), None, None, False),
        ((13, 4, 4), None, None, True),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), False),
        ((16, 5632), None, None, False),
        ((16, 5632), None, None, True),
        ((16, 5632), (13312, 1), (13312, 1), False),
        ((4, 4, 5632), None, None, False),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), False),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ),
)
def test_zeros_infinilm(shape, input_strides, out_strides, inplace, dtype, device):
    if device == "musa" and dtype == torch.float64:
        pytest.skip("MUSA does not support float64 zeros_infinilm")

    input = _make_input(shape, input_strides, dtype=dtype, device=device)
    out = (
        input
        if inplace
        else empty_strided(shape, out_strides, dtype=dtype, device=device)
    )
    if not inplace:
        _fill_nonzero(out)

    return Payload(
        _zeros_infinilm,
        _torch_zeros_infinilm,
        (input, out),
        {},
        rtol=0,
        atol=0,
    )


def _make_input(shape, strides, *, dtype, device):
    if dtype.is_floating_point:
        return randn_strided(shape, strides, dtype=dtype, device=device)
    return randint_strided(1, 16, shape, strides, dtype=dtype, device=device)


def _fill_nonzero(tensor):
    if tensor.dtype.is_floating_point:
        tensor.fill_(1)
    else:
        tensor.fill_(1)


def _zeros_infinilm(input, out):
    infini.ops.zeros_infinilm(input, out, stream=get_stream(input.device))

    return out


def _torch_zeros_infinilm(input, out):
    out.zero_()

    return out
