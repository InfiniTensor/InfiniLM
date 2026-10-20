import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape",
    (
        (1, 7),
        (3, 11),
        (16, 512),
    ),
)
@pytest.mark.parametrize("scale", (1.0, 0.5, 1.7))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_scaled_softmax_infinilm(
    shape,
    scale,
    dtype,
    device,
    implementation_index,
    rtol,
    atol,
):
    input_tensor = randn_strided(shape, None, dtype=dtype, device=device)
    out = empty_strided(shape, None, dtype=dtype, device=device)

    return Payload(
        _scaled_softmax_infinilm,
        _torch_scaled_softmax,
        (input_tensor, out),
        {"scale": scale, "implementation_index": implementation_index},
        rtol=rtol,
        atol=atol,
    )


def _scaled_softmax_infinilm(input_tensor, out, *, scale, implementation_index):
    infini.ops.scaled_softmax_infinilm(
        input_tensor,
        scale,
        out,
        stream=get_stream(input_tensor.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_scaled_softmax(input_tensor, out, *, scale, implementation_index):
    del implementation_index

    result = torch.nn.functional.softmax(input_tensor.to(torch.float32) * scale, dim=-1)
    out.copy_(result.to(input_tensor.dtype))

    return out
