import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides",
    (
        ((13, 4), None, None),
        ((13, 4), (10, 1), (10, 1)),
        ((13, 4, 4), None, None),
        ((16, 5632), None, None),
        ((4, 4, 5632), None, None),
    ),
)
@pytest.mark.parametrize(
    ("input_dtype", "out_dtype", "rtol", "atol"),
    (
        (torch.float16, torch.float32, 1e-3, 1e-3),
        (torch.float32, torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, torch.float32, 1e-2, 5e-3),
        (torch.float32, torch.bfloat16, 1e-2, 5e-3),
        (torch.float16, torch.bfloat16, 1e-2, 5e-3),
        (torch.bfloat16, torch.float16, 1e-2, 5e-3),
    ),
)
def test_cast(
    shape,
    input_strides,
    out_strides,
    input_dtype,
    out_dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(shape, input_strides, dtype=input_dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=out_dtype, device=device)

    return Payload(
        _cast,
        _torch_cast,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _cast(input, out):
    infini.ops.cast(input, out, stream=get_stream(input.device))

    return out


def _torch_cast(input, out):
    out.copy_(input.to(out.dtype))

    return out
