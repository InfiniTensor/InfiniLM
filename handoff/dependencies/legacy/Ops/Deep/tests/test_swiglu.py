import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, rand_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, gate_strides, out_strides",
    (
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4, 4), None, None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
        ((16, 5632), None, None, None),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
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
    ),
)
def test_swiglu(
    shape,
    input_strides,
    gate_strides,
    out_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input = rand_strided(shape, input_strides, dtype=dtype, device=device)
    gate = rand_strided(shape, gate_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _swiglu(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_swiglu,
        (input, gate, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _swiglu(input, gate, out, implementation_index=0):
    infini.ops.swiglu(
        input,
        gate,
        out,
        implementation_index=implementation_index,
        stream=get_stream(input.device),
    )

    return out


def _torch_swiglu(input, gate, out):
    swish_x = gate * torch.sigmoid(gate)

    return torch.mul(input, swish_x, out=out)
