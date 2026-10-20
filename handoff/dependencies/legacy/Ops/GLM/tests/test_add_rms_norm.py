import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (input_shape, weight_shape, input_strides, residual_strides, weight_strides, out_strides);
# input/residual/residual_out share input_shape.
_TEST_CASES = (
    ((1, 4), (4,), None, None, None, None),
    ((2, 4), (4,), None, None, None, None),
    ((2, 2, 4), (4,), None, None, None, None),
    ((2, 2, 4), (4,), (12, 8, 1), (12, 8, 1), None, (12, 8, 1)),
    ((16, 2048), (2048,), None, None, None, None),
    ((16, 2048), (2048,), (4096, 1), (4096, 1), None, (4096, 1)),
    ((15, 3584), (3584,), None, None, None, None),
    ((4, 4, 2048), (2048,), None, None, None, None),
    ((4, 4, 2048), (2048,), (2048, 8192, 1), (2048, 8192, 1), None, (2048, 8192, 1)),
    (
        (4, 4, 2048),
        (2048,),
        (16384, 4096, 1),
        (16384, 4096, 1),
        None,
        (16384, 4096, 1),
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("check_output", ("out", "residual_out"))
@pytest.mark.parametrize(
    "input_shape, weight_shape, input_strides, residual_strides, weight_strides, out_strides",
    _TEST_CASES,
)
@pytest.mark.parametrize("eps", (1e-6, 1e-5))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_add_rms_norm(
    check_output,
    input_shape,
    weight_shape,
    input_strides,
    residual_strides,
    weight_strides,
    out_strides,
    eps,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(input_shape, input_strides, dtype=dtype, device=device)
    residual = randn_strided(input_shape, residual_strides, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, weight_strides, dtype=dtype, device=device)
    residual_out = empty_strided(input_shape, input_strides, dtype=dtype, device=device)
    out = empty_strided(input_shape, out_strides, dtype=dtype, device=device)

    if check_output == "out":
        func = _add_rms_norm_out
        ref = _torch_add_rms_norm_out
    else:
        func = _add_rms_norm_residual_out
        ref = _torch_add_rms_norm_residual_out

    return Payload(
        lambda *args, **kwargs: func(
            *args, **kwargs, implementation_index=implementation_index
        ),
        ref,
        (input, residual, weight),
        {"eps": eps, "out": out, "residual_out": residual_out},
        rtol=rtol,
        atol=atol,
    )


def _add_rms_norm(
    input,
    residual,
    weight,
    *,
    eps=1e-6,
    out=None,
    residual_out=None,
    implementation_index=0,
):
    infini.ops.add_rms_norm(
        input,
        residual,
        weight,
        eps,
        out,
        residual_out,
        implementation_index=implementation_index,
        stream=get_stream(input.device),
    )


def _add_rms_norm_out(
    input,
    residual,
    weight,
    *,
    eps=1e-6,
    out=None,
    residual_out=None,
    implementation_index=0,
):
    _add_rms_norm(
        input,
        residual,
        weight,
        eps=eps,
        out=out,
        residual_out=residual_out,
        implementation_index=implementation_index,
    )

    return out


def _add_rms_norm_residual_out(
    input,
    residual,
    weight,
    *,
    eps=1e-6,
    out=None,
    residual_out=None,
    implementation_index=0,
):
    _add_rms_norm(
        input,
        residual,
        weight,
        eps=eps,
        out=out,
        residual_out=residual_out,
        implementation_index=implementation_index,
    )

    return residual_out


def _torch_add_rms_norm(
    input, residual, weight, *, eps=1e-6, out=None, residual_out=None
):
    """Reference aligned with vLLM `fused_add_rms_norm` (ignoring `variance_size`)."""
    orig_dtype = input.dtype
    x = input.to(torch.float32)
    x = x + residual.to(torch.float32)
    add_result = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    normalized_result = x.to(orig_dtype)

    if out is not None:
        out.copy_(normalized_result)
    else:
        out = normalized_result

    if residual_out is not None:
        residual_out.copy_(add_result)
    else:
        residual_out = add_result

    return out, residual_out


def _torch_add_rms_norm_out(
    input, residual, weight, *, eps=1e-6, out=None, residual_out=None
):
    out, _ = _torch_add_rms_norm(
        input, residual, weight, eps=eps, out=out, residual_out=residual_out
    )

    return out


def _torch_add_rms_norm_residual_out(
    input, residual, weight, *, eps=1e-6, out=None, residual_out=None
):
    _, residual_out = _torch_add_rms_norm(
        input, residual, weight, eps=eps, out=out, residual_out=residual_out
    )

    return residual_out
