import pytest
import torch

import infini.ops

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_SHAPE_CASES = (
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4, 4), None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
    ((16, 5632), None, None),
    ((4, 4, 5632), None, None),
)

_FLOAT_DTYPE_CASES = (
    (torch.float32, 1e-6, 1e-6),
    (torch.float16, 1e-3, 1e-3),
    (torch.bfloat16, 1e-2, 5e-3),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape, input_strides, out_strides", _SHAPE_CASES)
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _FLOAT_DTYPE_CASES)
def test_abs(
    shape,
    input_strides,
    out_strides,
    dtype,
    device,
    implementation_index,
    rtol,
    atol,
):
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda input, out: _abs(input, out, implementation_index),
        _torch_abs,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _abs(input, out, implementation_index):
    infini.ops.abs(
        input,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_abs(input, out):
    out.copy_(torch.abs(input))

    return out


@pytest.mark.smoke
def test_abs_torch_backend_uses_handle_stream(device):
    platform = {"cuda": "nvidia", "npu": "ascend"}.get(device)
    pytorch_slot = 8

    if platform is None:
        pytest.skip("The stream regression requires CUDA or NPU")
    if not infini.ops.Abs.active_implementation_indices(platform):
        pytest.skip(f"The stream regression requires the {platform} backend")
    if pytorch_slot not in infini.ops.Abs.active_implementation_indices(device):
        pytest.skip("The PyTorch backend is not active")

    input = torch.full((4096,), -1.0, device=device)
    out = torch.full_like(input, torch.nan)
    accelerator = getattr(torch, device)
    stream = accelerator.Stream()
    raw_stream = getattr(stream, f"{device}_stream")

    def call_abs():
        infini.ops.abs(
            input,
            out,
            stream=raw_stream,
            implementation_index=pytorch_slot,
        )

    try:
        call_abs()
        stream.synchronize()
        out.fill_(torch.nan)
        accelerator.synchronize()

        if device == "cuda":
            with accelerator.stream(stream):
                accelerator._sleep(50_000_000)
            call_abs()

            default_stream = accelerator.default_stream()
            with accelerator.stream(default_stream):
                snapshot = out.clone()
        else:
            lhs = torch.randn((4096, 4096), dtype=torch.float16, device=device)
            rhs = torch.randn((4096, 4096), dtype=torch.float16, device=device)
            busy_out = torch.empty_like(lhs)
            producer = accelerator.Stream()
            gate = accelerator.Event()
            accelerator.synchronize()

            with accelerator.stream(producer):
                for _ in range(32):
                    torch.mm(lhs, rhs, out=busy_out)
                gate.record()

            assert not producer.query()
            stream.wait_event(gate)
            call_abs()

            default_stream = accelerator.default_stream()
            with accelerator.stream(default_stream):
                snapshot = out.clone()
        default_stream.synchronize()
        assert torch.isnan(snapshot).all()

        stream.synchronize()
        torch.testing.assert_close(out, input.abs())
    finally:
        accelerator.synchronize()


@pytest.mark.smoke
def test_abs_torch_backend_accepts_null_stream(device):
    pytorch_slot = 8

    if device != "npu":
        pytest.skip("The null-stream regression requires NPU")
    if not infini.ops.Abs.active_implementation_indices("ascend"):
        pytest.skip("The null-stream regression requires the Ascend backend")
    if pytorch_slot not in infini.ops.Abs.active_implementation_indices(device):
        pytest.skip("The PyTorch backend is not active")

    input = torch.full((4096,), -1.0, device=device)
    out = torch.full_like(input, torch.nan)

    infini.ops.abs(
        input,
        out,
        stream=0,
        implementation_index=pytorch_slot,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(out, input.abs())
