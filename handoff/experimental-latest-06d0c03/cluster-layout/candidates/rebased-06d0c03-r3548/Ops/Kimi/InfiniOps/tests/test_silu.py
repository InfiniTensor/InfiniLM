import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides",
    (
        ((2, 4), None, None),
        ((128, 64), None, None),
        ((2, 4, 8), None, None),
        ((4, 48, 6), None, None),
        ((1, 2048), (4096, 1), (4096, 1)),
        ((8, 16, 32), None, None),
        ((16, 5632), None, None),
        ((4, 4, 5632), None, None),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_silu(
    shape,
    input_strides,
    out_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _silu(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_silu,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _silu(input, out, implementation_index=0):
    infini.ops.silu(
        input,
        out,
        implementation_index=implementation_index,
        stream=get_stream(input.device),
    )

    return out


def _torch_silu(input, out):
    out.copy_(input * torch.sigmoid(input))

    return out
