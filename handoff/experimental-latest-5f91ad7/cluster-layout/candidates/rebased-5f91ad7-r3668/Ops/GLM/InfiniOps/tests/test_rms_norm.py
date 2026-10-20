import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, weight_shape, input_strides, weight_strides, out_strides",
    (
        ((1, 64), (64,), None, None, None),
        ((2, 128), (128,), None, None, None),
        ((4, 48, 64), (64,), None, None, None),
        ((2, 4, 2048), (2048,), None, None, None),
        ((1, 64), (64,), (64, 1), (1,), (64, 1)),
        ((4, 48, 64), (64,), (3072, 64, 1), (1,), (3072, 64, 1)),
    ),
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
def test_rms_norm(
    input_shape,
    weight_shape,
    input_strides,
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
    weight = randn_strided(weight_shape, weight_strides, dtype=dtype, device=device)
    out = empty_strided(input_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _rms_norm(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_rms_norm,
        (input, weight),
        {"eps": eps, "out": out},
        rtol=rtol,
        atol=atol,
    )


def test_rms_norm_non_default_stream(device, implementation_index):
    if device == "cuda":
        accelerator = torch.cuda
        stream_attribute = "cuda_stream"
    elif device == "musa":
        accelerator = torch.musa
        stream_attribute = "musa_stream"
    elif device == "mlu":
        accelerator = torch.mlu
        stream_attribute = "mlu_stream"
    else:
        pytest.skip("non-default streams require an accelerator backend")

    input = torch.randn((32, 128), dtype=torch.float16, device=device)
    weight = torch.randn((128,), dtype=torch.float16, device=device)
    out = torch.zeros_like(input)
    expected = _torch_rms_norm(input, weight, out=torch.empty_like(out)).cpu()
    accelerator.synchronize()

    stream = accelerator.Stream()
    stream.wait_stream(accelerator.current_stream())
    stream_ptr = getattr(stream, stream_attribute)

    accelerator._sleep(50_000_000)
    try:
        infini.ops.rms_norm(
            input,
            weight,
            1e-6,
            out,
            implementation_index=implementation_index,
            stream=stream_ptr,
        )
        stream.synchronize()
        with accelerator.stream(stream):
            actual = out.cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    finally:
        accelerator.synchronize()


def _rms_norm(input, weight, *, eps=1e-6, out=None, implementation_index=0):
    infini.ops.rms_norm(
        input,
        weight,
        eps,
        out,
        implementation_index=implementation_index,
        stream=get_stream(input.device),
    )

    return out


def _torch_rms_norm(input, weight, *, eps=1e-6, out=None):
    # Fallback for `torch<2.3`: `rms_norm = (x / sqrt(mean(x^2) + eps)) * weight`.
    def _fallback(input, _normalized_shape, weight, *, eps=1e-6):
        rms = torch.sqrt(torch.mean(input * input, dim=-1, keepdim=True) + eps)

        return (input / rms) * weight

    rms_norm_fn = getattr(torch.nn.functional, "rms_norm", _fallback)

    result = rms_norm_fn(input, input.shape[-1:], weight=weight, eps=eps)

    if out is not None:
        out.copy_(result)
    else:
        out = result

    return out
