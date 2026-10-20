import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randint_strided


if not hasattr(infini.ops, "CutlassScaledMm"):
    pytest.skip(
        "`CutlassScaledMm` is not available on this platform", allow_module_level=True
    )


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("m, n, k", ((16, 32, 64), (64, 128, 128)))
@pytest.mark.parametrize(
    "per_token, per_channel",
    ((False, False), (False, True), (True, False), (True, True)),
)
@pytest.mark.parametrize("has_bias", (False, True))
@pytest.mark.parametrize("padded", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-2, 3e-1),
        (torch.bfloat16, 1e-2, 3e-1),
    ),
)
def test_cutlass_scaled_mm(
    m,
    n,
    k,
    per_token,
    per_channel,
    has_bias,
    padded,
    dtype,
    device,
    implementation_index,
    rtol,
    atol,
):
    a_strides = (k + 16, 1) if padded else None
    b_strides = (1, k + 16) if padded else (1, k)
    out_strides = (n + 16, 1) if padded else None
    a = randint_strided(-4, 5, (m, k), a_strides, dtype=torch.int8, device=device)
    b = randint_strided(-4, 5, (k, n), b_strides, dtype=torch.int8, device=device)

    scale_a_shape = (m, 1) if per_token else (1,)
    scale_b_shape = (1, n) if per_channel else (1,)
    scale_a = torch.rand(scale_a_shape, dtype=torch.float32, device=device)
    scale_b = torch.rand(scale_b_shape, dtype=torch.float32, device=device)
    bias = torch.rand((n,), dtype=dtype, device=device) if has_bias else None
    out = empty_strided((m, n), out_strides, dtype=dtype, device=device)

    return Payload(
        _cutlass_scaled_mm,
        _torch_cutlass_scaled_mm,
        (a, b, scale_a, scale_b, bias, dtype, out),
        {"implementation_index": implementation_index},
        rtol=rtol,
        atol=atol,
    )


def _cutlass_scaled_mm(
    a,
    b,
    scale_a,
    scale_b,
    bias,
    out_dtype,
    out,
    *,
    implementation_index,
):
    infini.ops.cutlass_scaled_mm(
        a,
        b,
        scale_a,
        scale_b,
        bias,
        out_dtype,
        out,
        stream=get_stream(a.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_cutlass_scaled_mm(
    a,
    b,
    scale_a,
    scale_b,
    bias,
    out_dtype,
    out,
    *,
    implementation_index,
):
    del implementation_index

    result = torch.matmul(a.float(), b.float()) * scale_a * scale_b

    if bias is not None:
        result = result + bias.float()

    out.copy_(result.to(out_dtype))

    return out


def test_cutlass_scaled_mm_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    m, n, k = 16, 32, 64
    a = torch.randint(-4, 5, (m, k), dtype=torch.int8, device=device)
    b = torch.randint(-4, 5, (n, k), dtype=torch.int8, device=device).t()
    scale_a = torch.rand((m, 1), dtype=torch.float32, device=device)
    scale_b = torch.rand((1, n), dtype=torch.float32, device=device)
    out = torch.empty((m, n), dtype=torch.float16, device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        _cutlass_scaled_mm(
            a,
            b,
            scale_a,
            scale_b,
            None,
            out.dtype,
            out,
            implementation_index=implementation_index,
        )

    stream.synchronize()
    expected = torch.matmul(a.float(), b.float()) * scale_a * scale_b
    torch.testing.assert_close(out, expected.to(out.dtype), rtol=1e-2, atol=3e-1)


def test_cutlass_scaled_mm_multi_gpu_device_guard(device, implementation_index):
    if device != "cuda" or torch.cuda.device_count() < 2:
        pytest.skip("multi-GPU device guard test requires two NVIDIA GPUs")

    original_device = torch.cuda.current_device()

    try:
        torch.cuda.set_device(0)
        target_device = torch.device("cuda:1")
        m, n, k = 16, 32, 64
        a = torch.randint(-4, 5, (m, k), dtype=torch.int8, device=target_device)
        b = torch.randint(-4, 5, (n, k), dtype=torch.int8, device=target_device).t()
        scale_a = torch.rand((m, 1), dtype=torch.float32, device=target_device)
        scale_b = torch.rand((1, n), dtype=torch.float32, device=target_device)
        out = torch.empty((m, n), dtype=torch.float16, device=target_device)
        stream = torch.cuda.Stream(device=target_device)
        stream.wait_stream(torch.cuda.current_stream(target_device))

        infini.ops.cutlass_scaled_mm(
            a,
            b,
            scale_a,
            scale_b,
            None,
            out.dtype,
            out,
            stream=stream.cuda_stream,
            implementation_index=implementation_index,
        )

        assert torch.cuda.current_device() == 0
        stream.synchronize()
        expected = torch.matmul(a.float(), b.float()) * scale_a * scale_b
        torch.testing.assert_close(out, expected.to(out.dtype), rtol=1e-2, atol=3e-1)
    finally:
        torch.cuda.set_device(original_device)
