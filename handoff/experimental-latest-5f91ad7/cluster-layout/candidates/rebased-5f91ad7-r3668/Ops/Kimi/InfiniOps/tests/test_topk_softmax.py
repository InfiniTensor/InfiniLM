import subprocess
import sys
import textwrap

import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("renormalize", (False, True))
@pytest.mark.parametrize("has_bias", (False, True))
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64, torch.uint32))
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-6, 1e-6),
        (torch.bfloat16, 1e-6, 1e-6),
    ),
)
def test_topk_softmax(dtype, index_dtype, has_bias, renormalize, rtol, atol):
    if not torch.cuda.is_available():
        pytest.skip("`topk_softmax` requires the NVIDIA backend")

    gating_output = torch.tensor(
        (
            (1.25, -0.5, 0.75, 2.0, -1.0),
            (-0.25, 1.5, 0.5, -1.25, 2.25),
            (0.125, 0.75, 2.5, 1.0, -0.75),
        ),
        dtype=dtype,
        device="cuda",
    )
    bias = None
    if has_bias:
        bias = torch.tensor(
            (0.0, 0.75, -0.5, -1.0, 1.25),
            dtype=torch.float32,
            device=gating_output.device,
        )
    outputs = _make_outputs(gating_output, topk=2, index_dtype=index_dtype)

    result = infini.ops.topk_softmax(
        gating_output,
        bias,
        None,
        renormalize,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    assert result is None
    expected = _reference(gating_output, bias, None, 2, renormalize)
    torch.testing.assert_close(outputs[0], expected[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(outputs[1], expected[1].to(index_dtype), rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


def test_topk_softmax_bias_only_changes_selection():
    if not torch.cuda.is_available():
        pytest.skip("`topk_softmax` requires the NVIDIA backend")

    gating_output = torch.tensor(((3.0, 2.0, 1.0),), dtype=torch.float32, device="cuda")
    bias = torch.tensor((-4.0, 0.0, 3.0), dtype=torch.float32, device="cuda")
    outputs = _make_outputs(gating_output, topk=1, index_dtype=torch.int32)

    infini.ops.topk_softmax(
        gating_output,
        bias,
        None,
        False,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    unbiased_scores = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
    assert outputs[1].item() == 2
    torch.testing.assert_close(outputs[0], unbiased_scores[:, 2:3])


@pytest.mark.parametrize("bias_value", (float("nan"), float("-inf")))
def test_topk_softmax_nonfinite_bias_selects_valid_experts(bias_value):
    if not torch.cuda.is_available():
        pytest.skip("`topk_softmax` requires the NVIDIA backend")

    gating_output = torch.tensor(
        ((1.0, 2.0, 3.0, 4.0),), dtype=torch.float32, device="cuda"
    )
    bias = torch.full((4,), bias_value, dtype=torch.float32, device="cuda")
    first = _make_outputs(gating_output, topk=3, index_dtype=torch.int32)
    second = _make_outputs(gating_output, topk=3, index_dtype=torch.int32)

    for outputs in (first, second):
        infini.ops.topk_softmax(
            gating_output,
            bias,
            None,
            False,
            *outputs,
            stream=get_stream(gating_output.device),
        )

    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    assert first[1].min().item() >= 0
    assert first[1].max().item() < gating_output.size(1)
    assert torch.unique(first[1]).numel() == 3


def test_topk_softmax_padding_and_token_expert_indices():
    if not torch.cuda.is_available():
        pytest.skip("`topk_softmax` requires the NVIDIA backend")

    gating_output = torch.tensor(
        (
            (0.25, 2.0, -0.5, 1.0),
            (1.5, -0.25, 0.75, 0.0),
            (-1.0, 0.5, 2.25, 1.25),
        ),
        dtype=torch.float16,
        device="cuda",
    )
    is_padding = torch.tensor((False, True, False), dtype=torch.bool, device="cuda")
    outputs = _make_outputs(gating_output, topk=3, index_dtype=torch.int64)

    infini.ops.topk_softmax(
        gating_output,
        None,
        is_padding,
        False,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    expected = _reference(gating_output, None, is_padding, 3, False)
    torch.testing.assert_close(outputs[0], expected[0])
    torch.testing.assert_close(outputs[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


def test_topk_softmax_non_default_stream():
    if not torch.cuda.is_available():
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    gating_output = torch.randn((7, 13), dtype=torch.bfloat16, device="cuda")
    outputs = _make_outputs(gating_output, topk=4, index_dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    infini.ops.topk_softmax(
        gating_output,
        None,
        None,
        True,
        *outputs,
        stream=stream.cuda_stream,
    )

    stream.synchronize()
    expected = _reference(gating_output, None, None, 4, True)
    torch.testing.assert_close(outputs[0], expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        outputs[1], expected[1].to(outputs[1].dtype), rtol=0, atol=0
    )
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


def test_topk_softmax_tie_selects_smaller_expert_index():
    if not torch.cuda.is_available():
        pytest.skip("`topk_softmax` requires the NVIDIA backend")

    gating_output = torch.zeros((1, 4), dtype=torch.float32, device="cuda")
    outputs = _make_outputs(gating_output, topk=3, index_dtype=torch.int32)

    infini.ops.topk_softmax(
        gating_output,
        None,
        None,
        False,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    torch.testing.assert_close(
        outputs[1], torch.tensor(((0, 1, 2),), dtype=torch.int32, device="cuda")
    )


def test_topk_softmax_padding_uses_uint32_max_sentinel():
    if not torch.cuda.is_available():
        pytest.skip("`topk_softmax` requires the NVIDIA backend")

    gating_output = torch.tensor(((1.0, 2.0, 3.0),), dtype=torch.float32, device="cuda")
    is_padding = torch.tensor((True,), dtype=torch.bool, device="cuda")
    outputs = _make_outputs(gating_output, topk=2, index_dtype=torch.uint32)

    infini.ops.topk_softmax(
        gating_output,
        None,
        is_padding,
        False,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    expected = torch.full(
        (1, 2),
        torch.iinfo(torch.uint32).max,
        dtype=torch.uint32,
        device="cuda",
    )
    torch.testing.assert_close(outputs[1], expected, rtol=0, atol=0)


def test_topk_softmax_empty_tokens():
    if not torch.cuda.is_available():
        pytest.skip("`topk_softmax` requires the NVIDIA backend")

    gating_output = torch.empty((0, 4), dtype=torch.float16, device="cuda")
    outputs = _make_outputs(gating_output, topk=2, index_dtype=torch.int64)

    result = infini.ops.topk_softmax(
        gating_output,
        None,
        None,
        False,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    assert result is None
    assert all(output.shape == (0, 2) for output in outputs)


def test_topk_softmax_descriptor_reuses_matching_metadata():
    if not torch.cuda.is_available():
        pytest.skip("`topk_softmax` requires the NVIDIA backend")

    gating_output = torch.randn((3, 7), dtype=torch.float16, device="cuda")
    bias = torch.randn((7,), dtype=torch.float32, device="cuda")
    is_padding = torch.tensor((False, True, False), dtype=torch.bool, device="cuda")
    outputs = _make_outputs(gating_output, topk=2, index_dtype=torch.int64)
    operator = infini.ops.TopkSoftmax(gating_output, bias, is_padding, True, *outputs)
    reused_input = torch.randn_like(gating_output)
    reused_bias = torch.randn_like(bias)
    reused_padding = is_padding.clone()
    reused_outputs = tuple(torch.empty_like(output) for output in outputs)

    operator(reused_input, reused_bias, reused_padding, True, *reused_outputs)

    expected = _reference(reused_input, reused_bias, reused_padding, 2, True)
    torch.testing.assert_close(reused_outputs[0], expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(reused_outputs[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(reused_outputs[2], expected[2], rtol=0, atol=0)


def test_topk_softmax_multi_gpu_device_guard():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("device guard test requires two NVIDIA GPUs")

    original_device = torch.cuda.current_device()

    try:
        torch.cuda.set_device(0)
        target_device = torch.device("cuda:1")
        gating_output = torch.randn((5, 8), dtype=torch.bfloat16, device=target_device)
        outputs = _make_outputs(gating_output, topk=3, index_dtype=torch.int32)
        stream = torch.cuda.Stream(device=target_device)
        stream.wait_stream(torch.cuda.current_stream(target_device))

        infini.ops.topk_softmax(
            gating_output,
            None,
            None,
            False,
            *outputs,
            stream=stream.cuda_stream,
        )

        assert torch.cuda.current_device() == 0
        stream.synchronize()
        expected = _reference(gating_output, None, None, 3, False)
        torch.testing.assert_close(outputs[0], expected[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(outputs[1], expected[1].to(torch.int32))
        torch.testing.assert_close(outputs[2], expected[2])
    finally:
        torch.cuda.set_device(original_device)


def _make_outputs(gating_output, topk, index_dtype):
    shape = (gating_output.size(0), topk)
    topk_weights = torch.full(
        shape, torch.nan, dtype=torch.float32, device=gating_output.device
    )
    topk_indices = torch.full(
        shape, 2**31 - 1, dtype=index_dtype, device=gating_output.device
    )
    token_expert_indices = torch.full(
        shape, -1, dtype=torch.int32, device=gating_output.device
    )

    return topk_weights, topk_indices, token_expert_indices


def _reference(gating_output, bias, is_padding, topk, renormalize):
    scores = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
    selection_scores = scores if bias is None else scores + bias
    indices = torch.topk(selection_scores, topk, dim=-1).indices
    weights = scores.gather(1, indices)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)

    if is_padding is not None:
        indices = indices.masked_fill(is_padding.bool().unsqueeze(-1), -1)

    num_tokens = gating_output.size(0)
    token_expert_indices = torch.arange(
        topk, dtype=torch.int32, device=gating_output.device
    ).unsqueeze(0).expand(num_tokens, -1) * num_tokens + torch.arange(
        num_tokens, dtype=torch.int32, device=gating_output.device
    ).unsqueeze(-1)

    return weights, indices, token_expert_indices


_LINKED_IMPLEMENTATION_INDEX = 16


@pytest.mark.parametrize("renormalize", (False, True))
@pytest.mark.parametrize("has_bias", (False, True))
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64, torch.uint32))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_topk_softmax_linked(dtype, index_dtype, has_bias, renormalize, device):
    if device != "cuda":
        pytest.skip("linked `topk_softmax` requires the NVIDIA backend")
    if _LINKED_IMPLEMENTATION_INDEX not in (
        infini.ops.TopkSoftmax.active_implementation_indices("nvidia")
    ):
        pytest.skip("linked `topk_softmax` provider is not active")

    gating_output = torch.tensor(
        (
            (1.25, -0.5, 0.75, 2.0, -1.0),
            (-0.25, 1.5, 0.5, -1.25, 2.25),
            (0.125, 0.75, 2.5, 1.0, -0.75),
        ),
        dtype=dtype,
        device=device,
    )
    bias = None
    if has_bias:
        bias = torch.tensor(
            (0.0, 0.75, -0.5, -1.0, 1.25),
            dtype=torch.float32,
            device=device,
        )
    outputs = _make_outputs(gating_output, topk=2, index_dtype=index_dtype)

    infini.ops.topk_softmax(
        gating_output,
        bias,
        None,
        renormalize,
        *outputs,
        stream=get_stream(gating_output.device),
        implementation_index=_LINKED_IMPLEMENTATION_INDEX,
    )

    expected = _reference(gating_output, bias, None, 2, renormalize)
    torch.testing.assert_close(outputs[0], expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(outputs[1], expected[1].to(index_dtype), rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


def test_topk_softmax_linked_rejects_is_padding(device):
    if device != "cuda":
        pytest.skip("linked `topk_softmax` requires the NVIDIA backend")
    if _LINKED_IMPLEMENTATION_INDEX not in (
        infini.ops.TopkSoftmax.active_implementation_indices("nvidia")
    ):
        pytest.skip("linked `topk_softmax` provider is not active")

    result = subprocess.run(
        [sys.executable, "-c", _IS_PADDING_SCRIPT], capture_output=True, text=True
    )

    assert result.returncode != 0
    assert "does not support `is_padding`" in result.stderr


_IS_PADDING_SCRIPT = textwrap.dedent(
    """
    import infini.ops
    import torch


    gating_output = torch.randn((2, 4), dtype=torch.float16, device="cuda")
    is_padding = torch.zeros((2,), dtype=torch.bool, device="cuda")
    outputs = (
        torch.empty((2, 2), dtype=torch.float32, device="cuda"),
        torch.empty((2, 2), dtype=torch.int32, device="cuda"),
        torch.empty((2, 2), dtype=torch.int32, device="cuda"),
    )
    infini.ops.topk_softmax(
        gating_output,
        None,
        is_padding,
        False,
        *outputs,
        implementation_index=16,
    )
    """
)
