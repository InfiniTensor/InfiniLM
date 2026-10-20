import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape",
    (
        ((4, 64), (64, 32), (4, 32)),
        ((2, 128), (128, 256), (2, 256)),
        ((64,), (64,), ()),
        ((64,), (64, 32), (32,)),
        ((4, 64), (64,), (4,)),
        ((2, 4, 64), (2, 64, 32), (2, 4, 32)),
        ((4, 8, 128), (4, 128, 64), (4, 8, 64)),
        ((2, 1, 4, 64), (3, 64, 32), (2, 3, 4, 32)),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-2, 1e-2),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_matmul(
    a_shape,
    b_shape,
    c_shape,
    dtype,
    device,
    rtol,
    atol,
):
    a = randn_strided(a_shape, None, dtype=dtype, device=device)
    b = randn_strided(b_shape, None, dtype=dtype, device=device)

    c = empty_strided(c_shape, None, dtype=dtype, device=device)

    return Payload(
        _matmul,
        _torch_matmul,
        (a, b, c),
        {},
        rtol=rtol,
        atol=atol,
    )


def _matmul(a, b, c):
    infini.ops.matmul(a, b, c, stream=get_stream(a.device))

    return c


def _torch_matmul(a, b, c):
    result = torch.matmul(a.float(), b.float()).to(c.dtype)
    c.copy_(result)

    return c
