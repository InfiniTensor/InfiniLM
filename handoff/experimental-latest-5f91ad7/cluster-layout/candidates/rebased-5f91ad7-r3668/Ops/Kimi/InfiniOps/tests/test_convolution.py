import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream


_TEST_CASES = (
    pytest.param((1, 2, 7), (4, 2, 3), id="1d"),
    pytest.param((1, 2, 5, 6), (3, 2, 3, 3), id="2d"),
    pytest.param((1, 2, 4, 5, 6), (3, 2, 3, 3, 3), id="3d"),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("input_shape, weight_shape", _TEST_CASES)
@pytest.mark.parametrize("has_bias", (False, True))
def test_convolution(input_shape, weight_shape, has_bias, device):
    input = torch.randn(input_shape, device=device) * 0.01
    weight = torch.randn(weight_shape, device=device) * 0.01
    bias = torch.randn(weight_shape[0], device=device) * 0.01 if has_bias else None
    spatial_ndim = len(input_shape) - 2
    stride = (1,) * spatial_ndim
    padding = (1,) * spatial_ndim
    dilation = (1,) * spatial_ndim
    output_padding = (0,) * spatial_ndim
    groups = 1
    expected = torch.convolution(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        False,
        output_padding,
        groups,
    )
    out = torch.empty_like(expected)

    return Payload(
        lambda input, weight, bias, out: _convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            out,
        ),
        lambda input, weight, bias, out: torch.convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            output_padding,
            groups,
        ),
        (input, weight, bias, out),
        {},
    )


def _convolution(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    output_padding,
    groups,
    out,
):
    infini.ops.convolution(
        input,
        weight,
        bias,
        list(stride),
        list(padding),
        list(dilation),
        False,
        list(output_padding),
        groups,
        out,
        stream=get_stream(input.device),
    )

    return out
