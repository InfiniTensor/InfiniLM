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

_INT_DTYPES = (torch.int16, torch.int32, torch.int64)

_UINT_DTYPES = tuple(
    filter(None, (getattr(torch, f"uint{bits}", None) for bits in (16, 32, 64)))
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, other_strides, out_strides",
    (
        ((), None, None, None),
        ((0, 4), None, None, None),
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4), (0, 1), None, None),
        ((13, 4, 4), None, None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
        ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
        ((16, 5632), None, None, None),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
        ((13, 16, 2), (128, 4, 1), (0, 2, 1), (64, 4, 1)),
        ((13, 16, 2), (128, 4, 1), (2, 0, 1), (64, 4, 1)),
        ((4, 4, 5632), None, None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    )
    + tuple((dtype, 0, 0) for dtype in _INT_DTYPES + _UINT_DTYPES),
)
def test_mul(
    shape, input_strides, other_strides, out_strides, dtype, device, rtol, atol
):
    if device in ("mlu", "musa") and dtype in _UINT_DTYPES:
        pytest.skip(
            f"The `{device}` test cloning path does not support `uint16`, `uint32`, or `uint64`."
        )

    if dtype in _INT_DTYPES or dtype in _UINT_DTYPES:
        input = randint_strided(
            0, 100, shape, input_strides, dtype=dtype, device=device
        )
        other = randint_strided(
            0, 100, shape, other_strides, dtype=dtype, device=device
        )
    else:
        input = randn_strided(shape, input_strides, dtype=dtype, device=device)
        other = randn_strided(shape, other_strides, dtype=dtype, device=device)

    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(_mul, _torch_mul, (input, other, out), {}, rtol=rtol, atol=atol)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, other_shape, out_shape",
    (
        ((1, 151936), (1, 1), (1, 151936)),
        ((1, 1), (1, 151936), (1, 151936)),
        ((3, 1), (1, 5), (3, 5)),
        ((1, 5), (3, 1), (3, 5)),
        ((5,), (2, 3, 5), (2, 3, 5)),
        ((2, 3, 5), (5,), (2, 3, 5)),
        ((), (2, 3), (2, 3)),
        ((2, 3), (), (2, 3)),
        ((0, 4), (1, 4), (0, 4)),
        ((1, 4), (0, 4), (0, 4)),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_mul_broadcast(input_shape, other_shape, out_shape, dtype, device, rtol, atol):
    input = torch.randn(input_shape, dtype=dtype, device=device)
    other = torch.randn(other_shape, dtype=dtype, device=device)
    out = torch.empty(out_shape, dtype=dtype, device=device)

    return Payload(
        _mul,
        _torch_mul,
        (input, other, out),
        {},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_mul_broadcast_strided(dtype, device, rtol, atol):
    input = randn_strided((3, 1), (2, 1), dtype=dtype, device=device)
    other = randn_strided((1, 5), (10, 2), dtype=dtype, device=device)
    out = empty_strided((3, 5), (10, 2), dtype=dtype, device=device)

    return Payload(
        _mul,
        _torch_mul,
        (input, other, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _mul(input, other, out):
    infini.ops.mul(input, other, out, stream=get_stream(input.device))

    return out


def _torch_mul(input, other, out):
    if input.dtype in _UINT_DTYPES:
        input = input.to(torch.int64)

    if other.dtype in _UINT_DTYPES:
        other = other.to(torch.int64)

    res = torch.mul(input, other)
    out.copy_(res.to(out.dtype))

    return out
