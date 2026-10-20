import subprocess
import sys
import textwrap

import infini.ops
import pytest
import torch

from tests.utils import get_stream


_VLLM_IMPLEMENTATION_INDEX = 16


if not hasattr(infini.ops, "GroupedTopk"):
    pytest.skip(
        "`GroupedTopk` is not available on this platform", allow_module_level=True
    )


@pytest.mark.parametrize(
    (
        "scores_dtype",
        "bias_dtype",
        "shape",
        "num_expert_group",
        "topk_group",
        "topk",
        "renormalize",
        "routed_scaling_factor",
        "scoring_func",
    ),
    (
        (torch.float16, torch.float32, (1, 16), 8, 2, 2, False, 1.0, 0),
        (torch.bfloat16, torch.float32, (5, 128), 8, 2, 4, True, 1.0, 0),
        (torch.float32, torch.float16, (3, 256), 8, 4, 8, False, 2.5, 0),
        (torch.float16, torch.bfloat16, (7, 128), 8, 4, 8, True, 2.5, 1),
        (torch.bfloat16, torch.float16, (2, 384), 1, 1, 8, False, 1.0, 1),
        (torch.float32, torch.bfloat16, (4, 512), 1, 1, 22, True, 2.5, 1),
    ),
)
def test_grouped_topk(
    scores_dtype,
    bias_dtype,
    shape,
    num_expert_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    scoring_func,
    device,
):
    if device != "cuda":
        pytest.skip("`grouped_topk` requires the NVIDIA backend")

    scores, bias = _make_inputs(shape, scores_dtype, bias_dtype, scoring_func, device)
    original_scores = scores.clone()
    topk_values, topk_indices = _make_outputs(scores, topk)

    infini.ops.grouped_topk(
        scores,
        bias,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        scoring_func,
        topk_values,
        topk_indices,
        stream=get_stream(scores.device),
    )

    expected_values, expected_indices = _reference(
        scores,
        bias,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        scoring_func,
    )
    tolerance = 1e-6 if scores_dtype == torch.float32 else 3e-3
    torch.testing.assert_close(
        topk_values, expected_values, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(topk_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(scores, original_scores, rtol=0, atol=0)


@pytest.mark.parametrize(
    (
        "dtype",
        "shape",
        "num_expert_group",
        "topk_group",
        "topk",
        "renormalize",
        "routed_scaling_factor",
        "scoring_func",
    ),
    (
        (torch.float32, (2, 16), 4, 2, 4, False, 1.0, 0),
        (torch.float16, (3, 64), 8, 2, 4, True, 2.5, 0),
        (torch.bfloat16, (4, 128), 8, 4, 8, True, 1.5, 1),
    ),
)
def test_grouped_topk_matches_vllm_provider(
    dtype,
    shape,
    num_expert_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    scoring_func,
):
    _require_vllm_implementation()
    if not torch.cuda.is_available():
        pytest.skip("vLLM `grouped_topk` requires an NVIDIA device")

    logits = torch.arange(shape[0] * shape[1], device="cuda", dtype=torch.float32)
    logits = ((logits * 37) % 29 - 14).reshape(shape) / 8
    scores = logits.to(dtype)
    bias = (((torch.arange(shape[1], device="cuda") * 11) % 13) - 6).float() / 16
    original_scores = scores.clone()
    topk_values = torch.empty((shape[0], topk), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((shape[0], topk), dtype=torch.int32, device="cuda")

    routed_scores = scores if scoring_func == 0 else torch.sigmoid(scores)
    scores_with_bias = (routed_scores + bias.unsqueeze(0)).to(dtype)
    expected_values, expected_indices = torch.ops._moe_C.grouped_topk(
        routed_scores,
        scores_with_bias,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
    )

    infini.ops.grouped_topk(
        scores,
        bias,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        scoring_func,
        topk_values,
        topk_indices,
        stream=get_stream(scores.device),
        implementation_index=_VLLM_IMPLEMENTATION_INDEX,
    )

    torch.testing.assert_close(topk_values, expected_values.float(), rtol=0, atol=0)
    torch.testing.assert_close(topk_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(scores, original_scores, rtol=0, atol=0)


def test_grouped_topk_uses_vllm_provider_tie_semantics():
    _require_vllm_implementation()
    if not torch.cuda.is_available():
        pytest.skip("vLLM `grouped_topk` requires an NVIDIA device")

    scores = torch.tensor(
        [[0.25, 0.5, 0.25, 0.5, 0.75, 0.75, 0.125, 0.125]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    bias = torch.zeros(8, dtype=torch.float32, device="cuda")
    topk_values = torch.empty((1, 4), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((1, 4), dtype=torch.int32, device="cuda")
    expected_values, expected_indices = torch.ops._moe_C.grouped_topk(
        scores,
        (scores + bias.unsqueeze(0)).to(scores.dtype),
        2,
        2,
        4,
        False,
        1.0,
    )

    infini.ops.grouped_topk(
        scores,
        bias,
        2,
        2,
        4,
        False,
        1.0,
        0,
        topk_values,
        topk_indices,
        stream=get_stream(scores.device),
        implementation_index=_VLLM_IMPLEMENTATION_INDEX,
    )

    torch.testing.assert_close(topk_values, expected_values.float(), rtol=0, atol=0)
    torch.testing.assert_close(topk_indices, expected_indices, rtol=0, atol=0)


def test_grouped_topk_ties_prefer_smaller_expert_indices(device):
    if device != "cuda":
        pytest.skip("`grouped_topk` requires the NVIDIA backend")

    scores = torch.full((1, 8), 0.5, dtype=torch.float32, device=device)
    bias = torch.zeros(8, dtype=torch.float32, device=device)
    topk_values, topk_indices = _make_outputs(scores, topk=4)

    infini.ops.grouped_topk(
        scores,
        bias,
        2,
        2,
        4,
        False,
        1.0,
        0,
        topk_values,
        topk_indices,
        stream=get_stream(scores.device),
    )

    expected_indices = torch.tensor(((0, 1, 2, 3),), dtype=torch.int32, device=device)
    expected_values = torch.full((1, 4), 0.5, device=device)
    torch.testing.assert_close(topk_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(topk_values, expected_values, rtol=0, atol=0)


def test_grouped_topk_group_ties_prefer_smaller_group_ids(device):
    if device != "cuda":
        pytest.skip("`grouped_topk` requires the NVIDIA backend")

    scores = torch.tensor(
        ((0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8),),
        dtype=torch.float32,
        device=device,
    )
    bias = torch.zeros(8, dtype=torch.float32, device=device)
    topk_values, topk_indices = _make_outputs(scores, topk=2)

    infini.ops.grouped_topk(
        scores,
        bias,
        4,
        2,
        2,
        False,
        1.0,
        0,
        topk_values,
        topk_indices,
        stream=get_stream(scores.device),
    )

    expected_indices = torch.tensor(((0, 2),), dtype=torch.int32, device=device)
    expected_values = torch.full((1, 2), 0.9, device=device)
    torch.testing.assert_close(topk_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(topk_values, expected_values, rtol=0, atol=0)


@pytest.mark.parametrize("scoring_func", (0, 1))
def test_grouped_topk_excludes_nonfinite_experts(scoring_func, device):
    if device != "cuda":
        pytest.skip("`grouped_topk` requires the NVIDIA backend")

    scores = torch.tensor(
        ((torch.nan, torch.inf, -torch.inf, 0.8, 0.9, 0.7, -0.1, 0.3),),
        dtype=torch.float32,
        device=device,
    )
    bias = torch.zeros(8, dtype=torch.float32, device=device)
    topk_values, topk_indices = _make_outputs(scores, topk=4)

    infini.ops.grouped_topk(
        scores,
        bias,
        2,
        2,
        4,
        False,
        1.0,
        scoring_func,
        topk_values,
        topk_indices,
        stream=get_stream(scores.device),
    )

    expected_values, expected_indices = _reference(
        scores, bias, 2, 2, 4, False, 1.0, scoring_func
    )
    assert torch.isfinite(topk_values).all()
    torch.testing.assert_close(topk_values, expected_values, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(topk_indices, expected_indices, rtol=0, atol=0)


def test_grouped_topk_descriptor_reuses_matching_metadata(device):
    if device != "cuda":
        pytest.skip("`grouped_topk` requires the NVIDIA backend")

    scores, bias = _make_inputs(
        (4, 32), torch.bfloat16, torch.float32, scoring_func=1, device=device
    )
    outputs = _make_outputs(scores, topk=4)
    operator = infini.ops.GroupedTopk(scores, bias, 4, 2, 4, True, 2.5, 1, *outputs)
    reused_scores, reused_bias = _make_inputs(
        (4, 32), torch.bfloat16, torch.float32, scoring_func=1, device=device
    )
    reused_scores = reused_scores.flip(0).contiguous()
    reused_bias = reused_bias.flip(0).contiguous()
    reused_outputs = _make_outputs(reused_scores, topk=4)

    operator(reused_scores, reused_bias, 4, 2, 4, True, 2.5, 1, *reused_outputs)

    expected = _reference(reused_scores, reused_bias, 4, 2, 4, True, 2.5, 1)
    torch.testing.assert_close(reused_outputs[0], expected[0], rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(reused_outputs[1], expected[1], rtol=0, atol=0)


def test_grouped_topk_non_default_stream(device):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    scores, bias = _make_inputs(
        (8, 64), torch.float16, torch.float32, scoring_func=1, device=device
    )
    outputs = _make_outputs(scores, topk=4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    infini.ops.grouped_topk(
        scores,
        bias,
        8,
        2,
        4,
        True,
        1.0,
        1,
        *outputs,
        stream=stream.cuda_stream,
    )

    stream.synchronize()
    expected = _reference(scores, bias, 8, 2, 4, True, 1.0, 1)
    torch.testing.assert_close(outputs[0], expected[0], rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(outputs[1], expected[1], rtol=0, atol=0)


def test_grouped_topk_device_guard():
    if not infini.ops.GroupedTopk.active_implementation_indices("nvidia"):
        pytest.skip("device guard test requires the NVIDIA implementation")
    if torch.cuda.device_count() < 2:
        pytest.skip("device guard test requires two NVIDIA GPUs")

    original_device = torch.cuda.current_device()

    try:
        torch.cuda.set_device(0)
        target_device = torch.device("cuda:1")
        scores, bias = _make_inputs(
            (3, 16), torch.float16, torch.float32, 1, target_device
        )
        outputs = _make_outputs(scores, topk=4)
        stream = torch.cuda.Stream(device=target_device)
        stream.wait_stream(torch.cuda.current_stream(target_device))

        infini.ops.grouped_topk(
            scores,
            bias,
            4,
            2,
            4,
            True,
            1.0,
            1,
            *outputs,
            stream=stream.cuda_stream,
        )

        assert torch.cuda.current_device() == 0
        stream.synchronize()
        expected = _reference(scores, bias, 4, 2, 4, True, 1.0, 1)
        torch.testing.assert_close(outputs[0], expected[0], rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(outputs[1], expected[1], rtol=0, atol=0)
    finally:
        torch.cuda.set_device(original_device)


def test_grouped_topk_rejects_single_expert_groups():
    if not infini.ops.GroupedTopk.active_implementation_indices("nvidia"):
        pytest.skip("validation test requires the NVIDIA implementation")
    if not torch.cuda.is_available():
        pytest.skip("validation test requires an NVIDIA device")

    result = subprocess.run(
        [sys.executable, "-c", _SINGLE_EXPERT_GROUP_SCRIPT],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        pytest.skip("descriptor validation requires an assertions-enabled build")

    assert "requires at least two experts per group" in result.stderr


def _make_inputs(shape, scores_dtype, bias_dtype, scoring_func, device):
    logits = torch.linspace(-2.25, 2.75, shape[0] * shape[1], device=device)
    logits = logits.reshape(shape)
    logits = logits + torch.sin(logits * 7.0) * 0.07
    scores = logits.to(scores_dtype)
    if scoring_func == 0:
        scores = torch.softmax(scores.float(), dim=-1).to(scores_dtype)

    bias = torch.linspace(-0.2, 0.3, shape[1], device=device)
    bias = bias.roll(3).to(bias_dtype)

    return scores, bias


def _require_vllm_implementation():
    if (
        _VLLM_IMPLEMENTATION_INDEX
        not in infini.ops.GroupedTopk.active_implementation_indices("nvidia")
    ):
        pytest.skip("vLLM `grouped_topk` implementation is not available")


def _make_outputs(scores, topk):
    shape = (scores.size(0), topk)
    topk_values = torch.full(
        shape, torch.nan, dtype=torch.float32, device=scores.device
    )
    topk_indices = torch.full(shape, -1, dtype=torch.int32, device=scores.device)

    return topk_values, topk_indices


def _reference(
    scores,
    bias,
    num_expert_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    scoring_func,
):
    if scoring_func == 0:
        routed_scores = scores
    else:
        routed_scores = (0.5 * torch.tanh(0.5 * scores.float()) + 0.5).to(scores.dtype)

    selection_scores = (routed_scores + bias.to(scores.dtype).unsqueeze(0)).to(
        scores.dtype
    )
    experts_per_group = scores.size(1) // num_expert_group
    grouped_scores = selection_scores.reshape(
        scores.size(0), num_expert_group, experts_per_group
    )
    grouped_scores = grouped_scores.masked_fill(torch.isnan(grouped_scores), -torch.inf)
    group_scores = torch.topk(grouped_scores, 2, dim=-1).values.sum(dim=-1)
    group_scores = group_scores.to(scores.dtype)
    selected_groups = torch.argsort(group_scores, dim=-1, descending=True, stable=True)[
        :, :topk_group
    ]
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = group_mask.repeat_interleave(experts_per_group, dim=-1)
    selected_scores = selection_scores.masked_fill(
        ~expert_mask | ~torch.isfinite(scores), -torch.inf
    )
    topk_indices = torch.argsort(selected_scores, dim=-1, descending=True, stable=True)[
        :, :topk
    ].to(torch.int32)
    topk_values = routed_scores.gather(1, topk_indices.to(torch.int64)).float()

    if renormalize:
        topk_values = topk_values / (topk_values.sum(dim=-1, keepdim=True) + 1e-20)
    topk_values = topk_values * routed_scaling_factor

    return topk_values, topk_indices


_SINGLE_EXPERT_GROUP_SCRIPT = textwrap.dedent(
    r"""
    import infini.ops
    import torch


    scores = torch.ones((1, 2), dtype=torch.float32, device="cuda")
    bias = torch.zeros(2, dtype=torch.float32, device="cuda")
    topk_values = torch.empty((1, 1), dtype=torch.float32, device="cuda")
    topk_indices = torch.empty((1, 1), dtype=torch.int32, device="cuda")
    infini.ops.GroupedTopk(
        scores,
        bias,
        2,
        1,
        1,
        False,
        1.0,
        0,
        topk_values,
        topk_indices,
    )
    """
)
