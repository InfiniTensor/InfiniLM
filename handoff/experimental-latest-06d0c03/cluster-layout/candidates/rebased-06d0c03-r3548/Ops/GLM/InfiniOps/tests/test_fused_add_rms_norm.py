import infini.ops
import pytest
import torch

from tests.utils import clone_strided, get_stream, randn_strided


_TEST_CASES = (
    ((1, 4), None, None),
    ((2, 4), None, None),
    ((2, 2, 4), None, None),
    ((2, 2, 4), (16, 8, 1), None),
    ((2, 2, 2, 8), None, None),
    ((16, 2048), None, None),
    ((15, 3584), None, None),
    ((2, 32769), None, None),
    ((2, 3, 4, 128), (3072, 1024, 256, 1), (3840, 1280, 320, 1)),
)


@pytest.mark.parametrize(
    "shape, input_strides, residual_strides",
    _TEST_CASES,
)
@pytest.mark.parametrize("has_weight", (False, True))
@pytest.mark.parametrize("epsilon", (1e-6, 1e-5))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_fused_add_rms_norm(
    shape,
    input_strides,
    residual_strides,
    has_weight,
    epsilon,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    residual = randn_strided(shape, residual_strides, dtype=dtype, device=device)
    weight = torch.randn(shape[-1], dtype=dtype, device=device) if has_weight else None
    expected_input = clone_strided(input)
    expected_residual = clone_strided(residual)

    _torch_fused_add_rms_norm(
        expected_input,
        expected_residual,
        weight,
        epsilon,
    )
    result = infini.ops.fused_add_rms_norm(
        input,
        residual,
        weight,
        epsilon,
        implementation_index=implementation_index,
        stream=get_stream(input.device),
    )

    assert result is None
    torch.testing.assert_close(input, expected_input, rtol=rtol, atol=atol)
    torch.testing.assert_close(residual, expected_residual, rtol=rtol, atol=atol)


def _torch_fused_add_rms_norm(input, residual, weight, epsilon):
    summed = (input.to(torch.float32) + residual.to(torch.float32)).to(input.dtype)
    variance = summed.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    normalized = summed.to(torch.float32) * torch.rsqrt(variance + epsilon)

    if weight is not None:
        normalized *= weight.to(torch.float32)

    residual.copy_(summed)
    input.copy_(normalized.to(input.dtype))
