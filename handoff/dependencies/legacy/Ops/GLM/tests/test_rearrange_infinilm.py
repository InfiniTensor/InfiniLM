import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


def _row_major_strides(shape):
    stride = 1
    strides = [1]
    for dim in reversed(shape[1:]):
        stride *= dim
        strides.insert(0, stride)

    return tuple(strides)


def _column_major_strides(shape):
    stride = 1
    strides = [stride]
    for dim in shape[:-1]:
        stride *= dim
        strides.append(stride)

    return tuple(strides)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides",
    (
        ((100, 100), (1, 100), (100, 1)),
        ((4, 4), (1, 4), (4, 1)),
        ((4, 6, 64), (64, 4 * 64, 1), (6 * 64, 64, 1)),
        ((2000, 2000), (1, 2000), (2000, 1)),
        ((2001, 2001), (1, 2001), (2001, 1)),
        ((2, 2, 2, 4), (16, 8, 4, 1), (16, 8, 1, 2)),
        (
            (3, 4, 7, 53, 9),
            _row_major_strides((3, 4, 7, 53, 9)),
            _column_major_strides((3, 4, 7, 53, 9)),
        ),
        (
            (3, 4, 50, 50, 5, 7),
            _row_major_strides((3, 4, 50, 50, 5, 7)),
            _column_major_strides((3, 4, 50, 50, 5, 7)),
        ),
        ((15, 10752), (0, 1), (10752, 1)),
        ((2, 2, 2, 2, 2, 2), (4, 8, 16, 32, 64, 128), (64, 32, 16, 8, 4, 2)),
        ((8, 4, 20, 64), (5120, 64, 256, 1), None),
        ((8, 4, 20, 64), (5120, 64, 256, 1), (1048576, 262144, 64, 1)),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 0, 0),
        (torch.float16, 0, 0),
    ),
)
def test_rearrange_infinilm(
    shape, input_strides, out_strides, dtype, device, rtol, atol
):
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        _rearrange_infinilm,
        _torch_rearrange_infinilm,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _rearrange_infinilm(input, out):
    infini.ops.rearrange_infinilm(input, out, stream=get_stream(input.device))

    return out


def _torch_rearrange_infinilm(input, out):
    out.copy_(input.expand_as(out))

    return out
