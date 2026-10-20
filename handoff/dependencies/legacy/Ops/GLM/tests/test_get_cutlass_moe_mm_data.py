import subprocess
import sys
import textwrap

import infini.ops
import pytest

import torch
from tests.utils import get_stream


def test_get_cutlass_moe_mm_data_binding_exists():
    assert hasattr(infini.ops, "get_cutlass_moe_mm_data")
    assert hasattr(infini.ops, "GetCutlassMoeMmData")


def test_linked_get_cutlass_moe_mm_data_rejects_unsupported_gating():
    _require_linked_implementation()
    result = _run_unsupported_call(is_gated=False)

    assert result.returncode != 0
    assert "only supports `is_gated=true`" in result.stderr


def test_linked_get_cutlass_moe_mm_data_rejects_unsupported_architecture():
    _require_linked_implementation()

    if torch.cuda.get_device_capability()[0] >= 9:
        pytest.skip("linked vLLM implementation supports this CUDA architecture")

    result = _run_unsupported_call(is_gated=True)

    assert result.returncode != 0
    assert "requires compute capability 9.0 or newer" in result.stderr


@pytest.mark.parametrize("is_gated", (False, True))
def test_get_cutlass_moe_mm_data_problem_sizes(is_gated, device, implementation_index):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    topk_ids = (torch.arange(66, dtype=torch.int32) % 4).reshape(33, 2).to(device)
    num_experts = 4
    n = 64
    k = 128
    outputs = _make_outputs(topk_ids, num_experts)

    _get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        is_gated,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, is_gated, *outputs)


@pytest.mark.parametrize("numel", (64, 65, 513))
def test_get_cutlass_moe_mm_data_swap_threshold_and_large_input(
    numel, device, implementation_index
):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    topk_ids = (torch.arange(numel, dtype=torch.int32) % 4).reshape(numel, 1)
    topk_ids = topk_ids.to(device)
    num_experts = 4
    n = 64
    k = 128
    outputs = _make_outputs(topk_ids, num_experts)

    _get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        True,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, True, *outputs)


def test_get_cutlass_moe_mm_data_defaults_to_gated_and_swaps_small_problem(
    device, implementation_index
):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    topk_ids = torch.tensor(
        ((2, 0), (1, 2), (0, 3), (3, 1)), dtype=torch.int32, device=device
    )
    num_experts = 4
    n = 64
    k = 128
    outputs = _make_outputs(topk_ids, num_experts)
    _skip_unsupported_linked_impl(implementation_index, is_gated=True)

    infini.ops.get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        *outputs,
        stream=get_stream(topk_ids.device),
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, True, *outputs)


@pytest.mark.parametrize(
    "topk_ids_values",
    (
        ((0, 1), (2, 3), (0, 3), (2, 1)),
        tuple((0, 1) for _ in range(130)),
    ),
)
def test_get_cutlass_moe_mm_data_blockscale_offsets(
    topk_ids_values, device, implementation_index
):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    topk_ids = torch.tensor(topk_ids_values, dtype=torch.int32, device=device)
    num_experts = 4
    n = 80
    k = 192
    outputs = _make_outputs(topk_ids, num_experts, with_blockscale=True)

    _get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        True,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, True, *outputs)


def test_get_cutlass_moe_mm_data_invalid_expert_uses_vllm_sentinel(
    device, implementation_index
):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    topk_ids = torch.tensor(
        ((0, 1), (2, -1), (3, 0), (-1, 2)), dtype=torch.int32, device=device
    )
    num_experts = 4
    n = 64
    k = 128
    outputs = _make_outputs(topk_ids, num_experts)

    _get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        True,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, True, *outputs)


def test_get_cutlass_moe_mm_data_descriptor_reuses_matching_metadata(device):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    num_experts = 4
    n = 64
    k = 128
    topk_ids = torch.tensor(
        ((0, 1), (2, 3), (3, 0), (1, 2)), dtype=torch.int32, device=device
    )
    outputs = _make_outputs(topk_ids, num_experts)
    operator = infini.ops.GetCutlassMoeMmData(
        topk_ids,
        num_experts,
        n,
        k,
        False,
        *outputs,
        None,
    )
    reused_topk_ids = torch.tensor(
        ((3, 2), (1, 0), (0, 2), (3, 1)), dtype=torch.int32, device=device
    )
    reused_outputs = _make_outputs(reused_topk_ids, num_experts)

    result = operator(
        reused_topk_ids,
        num_experts,
        n,
        k,
        False,
        *reused_outputs,
        None,
    )

    assert result is None
    _assert_matches_reference(
        reused_topk_ids, num_experts, n, k, False, *reused_outputs
    )


@pytest.mark.parametrize(
    "metadata_change",
    (
        "topk_strides",
        "topk_dtype",
        "problem_sizes_shape",
        "blockscale_presence",
        "num_experts",
        "is_gated",
    ),
)
def test_get_cutlass_moe_mm_data_descriptor_rejects_changed_metadata(
    metadata_change, device
):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    assertion_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            _DESCRIPTOR_REUSE_SCRIPT,
            "assertions_enabled_probe",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    if assertion_probe.returncode == 0:
        pytest.skip("descriptor validation requires an assertions-enabled build")

    result = subprocess.run(
        [sys.executable, "-c", _DESCRIPTOR_REUSE_SCRIPT, metadata_change],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    expected_message = (
        "`GetCutlassMoeMmData` attributes changed after descriptor creation"
        if metadata_change in ("num_experts", "is_gated")
        else "`GetCutlassMoeMmData` tensor metadata differs from its descriptor"
    )
    assert expected_message in result.stderr


def test_get_cutlass_moe_mm_data_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    topk_ids = torch.tensor(
        ((0, 1), (2, 3), (0, 3), (2, 1)), dtype=torch.int32, device=device
    )
    num_experts = 4
    n = 64
    k = 128
    outputs = _make_outputs(topk_ids, num_experts)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    _skip_unsupported_linked_impl(implementation_index, is_gated=True)

    infini.ops.get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        True,
        *outputs,
        stream=stream.cuda_stream,
        implementation_index=implementation_index,
    )

    stream.synchronize()
    _assert_matches_reference(topk_ids, num_experts, n, k, True, *outputs)


def test_get_cutlass_moe_mm_data_device_guard():
    if not infini.ops.GetCutlassMoeMmData.active_implementation_indices("nvidia"):
        pytest.skip("device guard test requires the NVIDIA implementation")

    if torch.cuda.device_count() < 2:
        pytest.skip("device guard test requires two NVIDIA GPUs")

    original_device = torch.cuda.current_device()

    try:
        torch.cuda.set_device(0)
        target_device = torch.device("cuda:1")
        topk_ids = torch.tensor(
            ((0, 1), (2, 3), (0, 3), (2, 1)),
            dtype=torch.int32,
            device=target_device,
        )
        num_experts = 4
        n = 64
        k = 128
        outputs = _make_outputs(topk_ids, num_experts)
        stream = torch.cuda.Stream(device=target_device)
        stream.wait_stream(torch.cuda.current_stream(target_device))

        infini.ops.get_cutlass_moe_mm_data(
            topk_ids,
            num_experts,
            n,
            k,
            True,
            *outputs,
            stream=stream.cuda_stream,
        )

        assert torch.cuda.current_device() == 0
        stream.synchronize()
        _assert_matches_reference(topk_ids, num_experts, n, k, True, *outputs)
    finally:
        torch.cuda.set_device(original_device)


def _make_outputs(topk_ids, num_experts, *, with_blockscale=False):
    numel = topk_ids.numel()
    expert_offsets = torch.empty(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )
    problem_sizes1 = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=topk_ids.device
    )
    problem_sizes2 = torch.empty_like(problem_sizes1)
    input_permutation = torch.full(
        (numel,), -1, dtype=torch.int32, device=topk_ids.device
    )
    output_permutation = torch.full_like(input_permutation, -1)
    outputs = (
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
    )

    if with_blockscale:
        outputs += (torch.empty_like(expert_offsets),)

    return outputs


def _get_cutlass_moe_mm_data(
    topk_ids,
    num_experts,
    n,
    k,
    is_gated,
    *outputs,
    implementation_index,
):
    _skip_unsupported_linked_impl(implementation_index, is_gated=is_gated)
    infini.ops.get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        is_gated,
        *outputs,
        stream=get_stream(topk_ids.device),
        implementation_index=implementation_index,
    )


def _skip_unsupported_linked_impl(implementation_index, *, is_gated):
    if implementation_index != 16:
        return

    if not is_gated:
        pytest.skip("linked vLLM implementation requires `is_gated=true`")

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("linked vLLM implementation requires compute capability 9.0")


def _require_linked_implementation():
    if not torch.cuda.is_available():
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    if 16 not in infini.ops.GetCutlassMoeMmData.active_implementation_indices("nvidia"):
        pytest.skip("linked vLLM implementation is not active")


def _run_unsupported_call(*, is_gated):
    return subprocess.run(
        [sys.executable, "-c", _UNSUPPORTED_CALL_SCRIPT, str(is_gated)],
        capture_output=True,
        check=False,
        text=True,
    )


def _assert_matches_reference(
    topk_ids,
    num_experts,
    n,
    k,
    is_gated,
    expert_offsets,
    problem_sizes1,
    problem_sizes2,
    input_permutation,
    output_permutation,
    blockscale_offsets=None,
):
    flattened = topk_ids.flatten().to(torch.int64)
    valid = (flattened >= 0) & (flattened < num_experts)
    valid_experts = flattened[valid]
    counts = torch.bincount(valid_experts, minlength=num_experts)
    expected_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=counts.device), counts.cumsum(0))
    )
    swap_ab = blockscale_offsets is None and topk_ids.numel() <= 64
    n1 = 2 * n if is_gated else n
    if swap_ab:
        expected_problem_sizes1 = torch.stack(
            (torch.full_like(counts, n1), counts, torch.full_like(counts, k)),
            dim=1,
        )
        expected_problem_sizes2 = torch.stack(
            (torch.full_like(counts, k), counts, torch.full_like(counts, n)),
            dim=1,
        )
    else:
        expected_problem_sizes1 = torch.stack(
            (counts, torch.full_like(counts, n1), torch.full_like(counts, k)),
            dim=1,
        )
        expected_problem_sizes2 = torch.stack(
            (counts, torch.full_like(counts, k), torch.full_like(counts, n)),
            dim=1,
        )

    torch.testing.assert_close(expert_offsets, expected_offsets.to(torch.int32))
    torch.testing.assert_close(problem_sizes1, expected_problem_sizes1.to(torch.int32))
    torch.testing.assert_close(problem_sizes2, expected_problem_sizes2.to(torch.int32))

    output_values = output_permutation.to("cpu").tolist()
    input_values = input_permutation.to("cpu").tolist()
    offset_values = expected_offsets.to("cpu").tolist()
    flattened_values = flattened.to("cpu").tolist()
    valid_count = int(valid.sum().item())
    topk = topk_ids.size(1)

    assert sorted(
        output_values[index]
        for index, expert in enumerate(flattened_values)
        if 0 <= expert < num_experts
    ) == list(range(valid_count))

    for route, expert in enumerate(flattened_values):
        destination = output_values[route]
        if expert == -1:
            assert destination == valid_count
            continue

        assert offset_values[expert] <= destination < offset_values[expert + 1]
        assert input_values[destination] == route // topk

    if blockscale_offsets is not None:
        rounded_counts = (counts + 127) // 128 * 128
        expected_blockscale_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=counts.device),
                rounded_counts.cumsum(0),
            )
        )
        torch.testing.assert_close(
            blockscale_offsets, expected_blockscale_offsets.to(torch.int32)
        )


_DESCRIPTOR_REUSE_SCRIPT = textwrap.dedent(
    r"""
    import sys

    import infini.ops
    import torch


    metadata_change = sys.argv[1]
    num_experts = 4
    n = 64
    k = 128
    is_gated = False
    topk_ids = torch.tensor(((0, 1), (2, 3)), dtype=torch.int32, device="cuda")
    expert_offsets = torch.empty(5, dtype=torch.int32, device="cuda")
    problem_sizes1 = torch.empty((4, 3), dtype=torch.int32, device="cuda")
    problem_sizes2 = torch.empty_like(problem_sizes1)
    input_permutation = torch.empty(4, dtype=torch.int32, device="cuda")
    output_permutation = torch.empty_like(input_permutation)
    blockscale_offsets = torch.empty_like(expert_offsets)
    operator = infini.ops.GetCutlassMoeMmData(
        topk_ids,
        num_experts,
        n,
        k,
        is_gated,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        blockscale_offsets,
    )

    if metadata_change in ("assertions_enabled_probe", "num_experts"):
        num_experts += 1
    elif metadata_change == "is_gated":
        is_gated = True
    elif metadata_change == "topk_strides":
        topk_ids = topk_ids.T
    elif metadata_change == "topk_dtype":
        topk_ids = topk_ids.to(torch.int64)
    elif metadata_change == "problem_sizes_shape":
        problem_sizes1 = torch.empty((4, 4), dtype=torch.int32, device="cuda")

    args = (
        topk_ids,
        num_experts,
        n,
        k,
        is_gated,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
    )
    args += (
        None if metadata_change == "blockscale_presence" else blockscale_offsets,
    )

    operator(*args)
    torch.cuda.synchronize()
    """
)


_UNSUPPORTED_CALL_SCRIPT = textwrap.dedent(
    r"""
    import sys

    import infini.ops
    import torch


    is_gated = sys.argv[1] == "True"
    topk_ids = torch.tensor(((0, 1), (2, 3)), dtype=torch.int32, device="cuda")
    expert_offsets = torch.empty(5, dtype=torch.int32, device="cuda")
    problem_sizes1 = torch.empty((4, 3), dtype=torch.int32, device="cuda")
    problem_sizes2 = torch.empty_like(problem_sizes1)
    input_permutation = torch.empty(4, dtype=torch.int32, device="cuda")
    output_permutation = torch.empty_like(input_permutation)
    infini.ops.get_cutlass_moe_mm_data(
        topk_ids,
        4,
        64,
        128,
        is_gated,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        None,
        implementation_index=16,
    )
    """
)
