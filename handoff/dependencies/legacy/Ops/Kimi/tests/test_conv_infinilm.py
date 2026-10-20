import math

import infini.ops
import pytest
import torch
import torch.nn.functional as F

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_TEST_CASES = (
    ((32, 3, 4), (12, 4, 1), (32, 3, 5), (15, 5, 1), (1,), (1,), (1,)),
    ((1, 3, 4, 4), (48, 16, 4, 1), (2, 3, 3, 3), (27, 9, 3, 1), (1, 1), (1, 2), (2, 1)),
    (
        (32, 3, 32, 32),
        (32 * 32 * 3, 32 * 32, 32, 1),
        (64, 3, 5, 5),
        (75, 25, 5, 1),
        (2, 2),
        (2, 2),
        (1, 1),
    ),
    (
        (1, 1, 4, 4, 4),
        (64, 64, 16, 4, 1),
        (1, 1, 5, 5, 5),
        (125, 125, 25, 5, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    (
        (32, 3, 32, 32, 32),
        (32 * 32 * 32 * 3, 32 * 32 * 32, 32 * 32, 32, 1),
        (64, 3, 5, 5, 5),
        (375, 125, 25, 5, 1),
        (3, 2, 2),
        (4, 3, 3),
        (2, 2, 1),
    ),
)


def _infer_output_shape(input_shape, weight_shape, padding, stride, dilation):
    spatial = tuple(
        math.floor(
            (
                input_shape[i + 2]
                + 2 * padding[i]
                - dilation[i] * (weight_shape[i + 2] - 1)
                - 1
            )
            / stride[i]
            + 1
        )
        for i in range(len(padding))
    )

    return (input_shape[0], weight_shape[0]) + spatial


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, input_strides, weight_shape, weight_strides, padding, stride, dilation",
    _TEST_CASES,
)
@pytest.mark.parametrize("has_bias", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-3),
    ),
)
def test_conv_infinilm(
    input_shape,
    input_strides,
    weight_shape,
    weight_strides,
    padding,
    stride,
    dilation,
    has_bias,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(input_shape, input_strides, dtype=dtype, device=device) * 0.01
    weight = (
        randn_strided(weight_shape, weight_strides, dtype=dtype, device=device) * 0.01
    )
    bias = (
        randn_strided((weight_shape[0],), (1,), dtype=dtype, device=device) * 0.01
        if has_bias and weight_shape[0] > 1
        else None
    )
    out_shape = _infer_output_shape(
        input_shape, weight_shape, padding, stride, dilation
    )
    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    return Payload(
        lambda *args: _conv_infinilm(
            *args, padding=padding, stride=stride, dilation=dilation
        ),
        lambda *args: _torch_conv_infinilm(
            *args, padding=padding, stride=stride, dilation=dilation
        ),
        (input, weight, bias, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _conv_infinilm(input, weight, bias, out, padding, stride, dilation):
    infini.ops.conv_infinilm(
        input,
        weight,
        bias,
        list(padding),
        list(stride),
        list(dilation),
        1,
        out,
        stream=get_stream(input.device),
    )

    return out


def _torch_conv_infinilm(input, weight, bias, out, padding, stride, dilation):
    ndim = input.ndim - 2
    if ndim == 1:
        result = F.conv1d(
            input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation
        )
    elif ndim == 2:
        result = F.conv2d(
            input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation
        )
    else:
        result = F.conv3d(
            input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation
        )

    out.copy_(result)

    return out
