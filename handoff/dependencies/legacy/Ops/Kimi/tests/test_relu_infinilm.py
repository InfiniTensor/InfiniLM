import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, rand_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides, inplace",
    (
        ((1, 3), None, None, False),
        ((1, 3), None, None, True),
        ((3, 3), None, None, False),
        ((3, 3), (5, 1), (5, 1), False),
        ((32, 20, 512), None, None, False),
        ((32, 20, 512), None, None, True),
        ((33, 333, 333), None, None, False),
        ((32, 256, 112, 112), None, None, False),
        ((3, 3, 13, 9, 17), None, None, False),
        (
            (3, 3, 13, 9, 17),
            (19890, 6630, 510, 34, 1),
            (19890, 6630, 510, 34, 1),
            False,
        ),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-3, 1e-3),
    ),
)
def test_relu_infinilm(
    shape, input_strides, out_strides, inplace, dtype, device, rtol, atol
):
    input = rand_strided(shape, input_strides, dtype=dtype, device=device)
    input.mul_(2).sub_(1)
    out = (
        input
        if inplace
        else empty_strided(shape, out_strides, dtype=dtype, device=device)
    )

    return Payload(
        _relu_infinilm,
        _torch_relu_infinilm,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _relu_infinilm(input, out):
    infini.ops.relu_infinilm(input, out, stream=get_stream(input.device))

    return out


def _torch_relu_infinilm(input, out):
    result = torch.nn.functional.relu(input)
    out.copy_(result)

    return out
