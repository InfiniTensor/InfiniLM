import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shapes, input_strides, dim, out_shape",
    (
        # 2 inputs, dim=0
        (((4, 64), (4, 64)), None, 0, (8, 64)),
        # 2 inputs, dim=1
        (((4, 32), (4, 64)), None, 1, (4, 96)),
        # 2 inputs, dim=-1 (negative dim)
        (((4, 32), (4, 64)), None, -1, (4, 96)),
        # 3 inputs, dim=1
        (((4, 16), (4, 32), (4, 16)), None, 1, (4, 64)),
        # 2 inputs, dim=0, 3D
        (((2, 4, 64), (2, 4, 64)), None, 0, (4, 4, 64)),
        # 2 inputs, dim=2, 3D
        (((2, 4, 32), (2, 4, 64)), None, 2, (2, 4, 96)),
        # 4 inputs, dim=1
        (
            ((1, 1024), (1, 1024), (1, 1024), (1, 1024)),
            None,
            1,
            (1, 4096),
        ),
        # 1 input
        (((4, 64),), None, 0, (4, 64)),
        # Empty input dimension
        (((0, 4), (3, 4)), None, 0, (3, 4)),
        # Non-contiguous inputs.
        (
            ((2, 3, 4), (2, 3, 4)),
            ((4, 8, 1), (4, 8, 1)),
            1,
            (2, 6, 4),
        ),
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
def test_cat(shapes, input_strides, dim, out_shape, dtype, device, rtol, atol):
    if input_strides is None:
        input_strides = (None,) * len(shapes)

    inputs = [
        randn_strided(shape, strides, dtype=dtype, device=device)
        for shape, strides in zip(shapes, input_strides)
    ]
    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    return Payload(
        lambda *args: _cat(*args, dim=dim),
        lambda *args: _torch_cat(*args, dim=dim),
        (*inputs, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _cat(*args, dim):
    inputs = list(args[:-1])
    out = args[-1]

    infini.ops.cat(inputs, dim, out, stream=get_stream(inputs[0].device))

    return out


def _torch_cat(*args, dim):
    inputs = list(args[:-1])
    out = args[-1]

    result = torch.cat(inputs, dim=dim)
    out.copy_(result)

    return out
