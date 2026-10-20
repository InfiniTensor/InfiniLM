import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, dim, inplace",
    (
        ((4, 4), 0, True),
        ((4, 4), 0, False),
        ((12, 16, 512, 512), 0, True),
        ((12, 16, 512, 512), 0, False),
        ((12, 16, 512, 512), 1, True),
        ((12, 16, 512, 512), 1, False),
        ((12, 16, 512, 512), 2, True),
        ((12, 16, 512, 512), 2, False),
        ((12, 16, 512, 512), 3, True),
        ((12, 16, 512, 512), 3, False),
        ((1, 16, 512, 512), 0, True),
        ((1, 16, 512, 512), 0, False),
        ((1, 16, 512, 512), 1, True),
        ((1, 16, 512, 512), 1, False),
        ((1, 16, 512, 512), 2, True),
        ((1, 16, 512, 512), 2, False),
        ((1, 16, 512, 512), 3, True),
        ((1, 16, 512, 512), 3, False),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 3e-5),
        (torch.float16, 1e-2, 1e-3),
    ),
)
def test_softmax_infinilm(shape, dim, inplace, dtype, device, rtol, atol):
    input = randn_strided(shape, None, dtype=dtype, device=device)
    out = input if inplace else torch.empty_like(input)

    return Payload(
        lambda *args: _softmax_infinilm(*args, dim=dim),
        lambda *args: _torch_softmax_infinilm(*args, dim=dim),
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _softmax_infinilm(input, out, dim):
    infini.ops.softmax_infinilm(input, dim, None, out, stream=get_stream(input.device))

    return out


def _torch_softmax_infinilm(input, out, dim):
    out.copy_(torch.softmax(input, dim=dim))

    return out
