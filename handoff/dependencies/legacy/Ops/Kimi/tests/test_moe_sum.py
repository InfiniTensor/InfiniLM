import subprocess
import sys
import textwrap

import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "MoeSum"):
    pytest.skip("`MoeSum` is not available on this platform", allow_module_level=True)


@pytest.mark.parametrize("topk, hidden_size", ((1, 7), (2, 64), (4, 129)))
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_moe_sum(
    topk,
    hidden_size,
    dtype,
    device,
    implementation_index,
    rtol,
    atol,
):
    if device != "cuda":
        pytest.skip("moe_sum requires the NVIDIA backend")

    input = torch.randn((5, topk, hidden_size), dtype=dtype, device=device)
    out = torch.full(
        (input.size(0), hidden_size), torch.nan, dtype=dtype, device=device
    )

    result = infini.ops.moe_sum(
        input,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    assert result is None
    expected = input.float().sum(dim=1).to(dtype)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("topk_ids_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("has_expert_map", (False, True))
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_moe_sum_routing(
    topk_ids_dtype,
    has_expert_map,
    dtype,
    device,
    implementation_index,
    rtol,
    atol,
):
    if device != "cuda":
        pytest.skip("moe_sum requires the NVIDIA backend")

    input = torch.randn((3, 4, 17), dtype=dtype, device=device)
    topk_ids = torch.tensor(
        ((0, 1, -1, 2), (2, -1, 1, 0), (1, 2, 0, -1)),
        dtype=topk_ids_dtype,
        device=device,
    )
    expert_map = torch.tensor((0, -1, 1), dtype=torch.int32, device=device)
    out = torch.full((3, 17), torch.nan, dtype=dtype, device=device)

    result = infini.ops.moe_sum(
        input,
        topk_ids,
        expert_map if has_expert_map else None,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    assert result is None
    valid_ids = topk_ids.clamp_min(0).to(torch.int64)
    active = topk_ids >= 0
    if has_expert_map:
        active &= expert_map[valid_ids] >= 0
    expected = (input.float() * active.unsqueeze(-1)).sum(dim=1).to(dtype)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


def test_moe_sum_non_contiguous_positive_strides(device, implementation_index):
    if device != "cuda":
        pytest.skip("moe_sum requires the NVIDIA backend")

    input = torch.empty_strided(
        (3, 4, 17), (167, 40, 2), dtype=torch.float32, device=device
    )
    input.normal_()
    topk_ids = torch.empty_strided((3, 4), (11, 2), dtype=torch.int64, device=device)
    topk_ids.copy_(
        torch.tensor(
            ((0, 1, -1, 2), (2, -1, 1, 0), (1, 2, 0, -1)),
            dtype=topk_ids.dtype,
            device=device,
        )
    )
    output = torch.full((3, 17), torch.nan, dtype=input.dtype, device=device)

    infini.ops.moe_sum(
        input,
        topk_ids,
        None,
        output,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    expected = (input * (topk_ids >= 0).unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(output, expected)


def test_moe_sum_descriptor_reuses_metadata_with_fresh_tensors(
    device, implementation_index
):
    if device != "cuda":
        pytest.skip("moe_sum requires the NVIDIA backend")

    input = torch.randn((3, 2, 17), dtype=torch.float32, device=device)
    output = torch.empty((3, 17), dtype=input.dtype, device=device)
    op = infini.ops.MoeSum(input, output)
    fresh_input = torch.randn_like(input)
    fresh_output = torch.full_like(output, torch.nan)

    result = op(fresh_input, fresh_output)

    assert result is None
    torch.testing.assert_close(fresh_output, fresh_input.sum(dim=1))


@pytest.mark.parametrize(
    "mismatch",
    (
        "shape",
        "stride",
        "dtype",
        "device",
        "optional_presence",
        "optional_metadata",
    ),
)
def test_moe_sum_descriptor_rejects_metadata_mismatch(mismatch, device):
    if device != "cuda":
        pytest.skip("moe_sum requires the NVIDIA backend")

    mismatch_setup = {
        "shape": """
call_input = torch.randn((3, 2, 18), dtype=torch.float32, device="cuda")
call_output = torch.empty((3, 18), dtype=torch.float32, device="cuda")
op(call_input, call_output)
""",
        "stride": """
call_input = torch.empty_strided(
    (3, 2, 17), (50, 22, 1), dtype=torch.float32, device="cuda"
)
call_input.normal_()
op(call_input, torch.empty_like(output))
""",
        "dtype": """
call_input = input.to(torch.float16)
call_output = output.to(torch.float16)
op(call_input, call_output)
""",
        "device": """
op(torch.randn_like(input, device="cpu"), torch.empty_like(output, device="cpu"))
""",
        "optional_presence": """
topk_ids = torch.zeros((3, 2), dtype=torch.int32, device="cuda")
op_with_routing = infini.ops.MoeSum(input, topk_ids, None, output)
op_with_routing(torch.randn_like(input), torch.empty_like(output))
""",
        "optional_metadata": """
topk_ids = torch.zeros((3, 2), dtype=torch.int32, device="cuda")
op_with_routing = infini.ops.MoeSum(input, topk_ids, None, output)
call_topk_ids = topk_ids.to(torch.int64)
op_with_routing(
    torch.randn_like(input), call_topk_ids, None, torch.empty_like(output)
)
""",
    }[mismatch]
    code = (
        textwrap.dedent(
            """
        import infini.ops
        import torch

        input = torch.randn((3, 2, 17), dtype=torch.float32, device="cuda")
        output = torch.empty((3, 17), dtype=torch.float32, device="cuda")
        op = infini.ops.MoeSum(input, output)
        """
        )
        + textwrap.dedent(mismatch_setup)
        + "\ntorch.cuda.synchronize()\n"
    )

    completed = subprocess.run(
        (sys.executable, "-c", code),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "`MoeSum` call metadata must match descriptor" in completed.stderr


def test_moe_sum_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    input = torch.randn((5, 2, 64), dtype=torch.float16, device=device)
    out = torch.full((5, 64), torch.nan, dtype=input.dtype, device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    infini.ops.moe_sum(
        input,
        None,
        None,
        out,
        stream=stream.cuda_stream,
        implementation_index=implementation_index,
    )

    stream.synchronize()
    expected = input.float().sum(dim=1).to(input.dtype)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_moe_sum_multi_gpu_device_guard(device, implementation_index):
    if device != "cuda" or torch.cuda.device_count() < 2:
        pytest.skip("multi-GPU device guard test requires two NVIDIA GPUs")

    original_device = torch.cuda.current_device()

    try:
        torch.cuda.set_device(0)
        target_device = torch.device("cuda:1")
        input = torch.randn((5, 4, 64), dtype=torch.bfloat16, device=target_device)
        out = torch.full((5, 64), torch.nan, dtype=input.dtype, device=target_device)
        stream = torch.cuda.Stream(device=target_device)
        stream.wait_stream(torch.cuda.current_stream(target_device))

        infini.ops.moe_sum(
            input,
            None,
            None,
            out,
            stream=stream.cuda_stream,
            implementation_index=implementation_index,
        )

        assert torch.cuda.current_device() == 0
        stream.synchronize()
        expected = input.float().sum(dim=1).to(input.dtype)
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
    finally:
        torch.cuda.set_device(original_device)
