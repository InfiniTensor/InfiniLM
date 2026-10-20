import infini.ops
import pytest
import torch
import torch.nn.functional as F

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_TEST_CASES = (
    pytest.param(
        (1, 3, 7, 8),
        (200, 60, 8, 1),
        (4, 3, 3, 2),
        (24, 8, 2, 1),
        (2, 1),
        (1, 0),
        (1, 2),
        1,
        id="numeric-strided",
    ),
    pytest.param(
        (2, 4, 6, 7),
        (168, 42, 7, 1),
        (6, 2, 2, 3),
        (12, 6, 3, 1),
        (1, 1),
        "valid",
        (1, 1),
        2,
        id="valid-grouped",
    ),
    pytest.param(
        (1, 2, 5, 6),
        (60, 30, 6, 1),
        (4, 2, 2, 4),
        (16, 8, 4, 1),
        (1, 1),
        "same",
        (1, 1),
        1,
        id="same-asymmetric",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, input_strides, weight_shape, weight_strides, stride, padding, dilation, groups",
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
def test_conv2d(
    input_shape,
    input_strides,
    weight_shape,
    weight_strides,
    stride,
    padding,
    dilation,
    groups,
    has_bias,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(input_shape, input_strides, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, weight_strides, dtype=dtype, device=device)
    bias = (
        randn_strided((weight_shape[0],), (2,), dtype=dtype, device=device)
        if has_bias
        else None
    )
    input = input * 0.01
    weight = weight * 0.01
    bias = bias * 0.01 if bias is not None else None
    out_shape = F.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ).shape
    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    return Payload(
        lambda *args: _conv2d(
            *args,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        ),
        lambda input, weight, bias, out: F.conv2d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        ),
        (input, weight, bias, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _conv2d(input, weight, bias, out, stride, padding, dilation, groups):
    resolved_padding = padding if isinstance(padding, str) else list(padding)
    infini.ops.conv2d(
        input,
        weight,
        bias,
        list(stride),
        resolved_padding,
        list(dilation),
        groups,
        out,
        stream=get_stream(input.device),
    )

    return out
