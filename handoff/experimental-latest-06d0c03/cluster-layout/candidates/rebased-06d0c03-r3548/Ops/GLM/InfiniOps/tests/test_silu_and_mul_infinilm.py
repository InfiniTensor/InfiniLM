import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, output_shape",
    (
        ((2, 8), (2, 4)),
        ((1024, 1024), (1024, 512)),
        ((16, 8192), (16, 4096)),
        ((2, 128, 2048), (2, 128, 1024)),
        ((8, 1, 4096), (8, 1, 2048)),
        ((2, 4, 16, 256), (2, 4, 16, 128)),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_silu_and_mul_infinilm(input_shape, output_shape, dtype, device, rtol, atol):
    input = randn_strided(input_shape, None, dtype=dtype, device=device)
    out = empty_strided(output_shape, None, dtype=dtype, device=device)

    return Payload(
        _silu_and_mul_infinilm,
        _torch_silu_and_mul_infinilm,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _silu_and_mul_infinilm(input, out):
    infini.ops.silu_and_mul_infinilm(input, out, stream=get_stream(input.device))

    return out


def _torch_silu_and_mul_infinilm(input, out):
    hidden = input.shape[-1] // 2
    gate = input[..., :hidden]
    up = input[..., hidden:]
    out.copy_(torch.nn.functional.silu(gate) * up)

    return out
