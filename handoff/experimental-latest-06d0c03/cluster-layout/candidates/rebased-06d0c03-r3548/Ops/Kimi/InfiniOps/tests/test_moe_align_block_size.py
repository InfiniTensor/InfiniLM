import subprocess
import sys
import textwrap

import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "MoeAlignBlockSize"):
    pytest.skip(
        "`MoeAlignBlockSize` is not available on this platform",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "topk_ids_values, num_experts, block_size",
    (
        (((0, 2), (0, 2), (2, 0)), 4, 2),
        (((1, 1), (3, 0), (3, 3)), 5, 4),
        (((4, 1), (0, 4), (2, 1), (4, 2)), 5, 8),
        (((-1, 4), (2**31 - 1, 0), (2, -(2**31))), 4, 4),
    ),
)
def test_moe_align_block_size(
    topk_ids_values,
    num_experts,
    block_size,
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("`moe_align_block_size` requires the NVIDIA backend")

    topk_ids = torch.tensor(topk_ids_values, dtype=torch.int32, device=device)
    outputs = _make_outputs(topk_ids, num_experts, block_size)

    _moe_align_block_size(
        topk_ids,
        None,
        num_experts,
        block_size,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, None, num_experts, block_size, *outputs)


@pytest.mark.parametrize(
    "topk_ids_values, expert_map_values, num_experts, block_size",
    (
        (
            ((0, 1), (2, 3), (0, 2), (3, 1)),
            (0, -1, 1, -1),
            4,
            4,
        ),
        (
            ((-1, 0), (1, 2), (3, 4), (2**31 - 1, 1)),
            (0, 5, -1, 2),
            4,
            4,
        ),
    ),
)
def test_moe_align_block_size_expert_map(
    topk_ids_values,
    expert_map_values,
    num_experts,
    block_size,
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("`moe_align_block_size` requires the NVIDIA backend")

    topk_ids = torch.tensor(topk_ids_values, dtype=torch.int32, device=device)
    expert_map = torch.tensor(expert_map_values, dtype=torch.int32, device=device)
    outputs = _make_outputs(topk_ids, num_experts, block_size)

    _moe_align_block_size(
        topk_ids,
        expert_map,
        num_experts,
        block_size,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, expert_map, num_experts, block_size, *outputs)


def test_moe_align_block_size_large_sparse_case(device, implementation_index):
    if device != "cuda":
        pytest.skip("`moe_align_block_size` requires the NVIDIA backend")

    num_experts = 257
    block_size = 128
    routed_experts = torch.arange(390, dtype=torch.int32, device=device) % 3
    routed_experts[0] = 256
    topk_ids = routed_experts.reshape(130, 3)
    outputs = _make_outputs(topk_ids, num_experts, block_size)

    _moe_align_block_size(
        topk_ids,
        None,
        num_experts,
        block_size,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, None, num_experts, block_size, *outputs)


def test_moe_align_block_size_is_deterministic(device, implementation_index):
    if device != "cuda":
        pytest.skip("`moe_align_block_size` requires the NVIDIA backend")

    num_experts = 7
    block_size = 32
    topk_ids = (torch.arange(2048, dtype=torch.int32, device=device) % 7).reshape(
        512, 4
    )
    results = []

    for _ in range(5):
        outputs = _make_outputs(topk_ids, num_experts, block_size)
        _moe_align_block_size(
            topk_ids,
            None,
            num_experts,
            block_size,
            *outputs,
            implementation_index=implementation_index,
        )
        results.append(tuple(output.clone() for output in outputs))

    torch.cuda.synchronize(topk_ids.device)
    expected = results[0]

    for result in results[1:]:
        assert all(
            torch.equal(actual, reference)
            for actual, reference in zip(result, expected)
        )


def test_moe_align_block_size_descriptor_reuses_matching_metadata(device):
    if device != "cuda":
        pytest.skip("`moe_align_block_size` requires the NVIDIA backend")

    num_experts = 4
    block_size = 2
    topk_ids = torch.tensor(((0, 1), (2, 3)), dtype=torch.int32, device=device)
    expert_map = torch.arange(num_experts, dtype=torch.int32, device=device)
    outputs = _make_outputs(topk_ids, num_experts, block_size)
    operator = infini.ops.MoeAlignBlockSize(
        topk_ids,
        expert_map,
        num_experts,
        block_size,
        *outputs,
    )
    reused_topk_ids = topk_ids.clone()
    reused_expert_map = expert_map.clone()
    reused_outputs = tuple(torch.empty_like(output) for output in outputs)

    operator(
        reused_topk_ids,
        reused_expert_map,
        num_experts,
        block_size,
        *reused_outputs,
    )

    _assert_matches_reference(
        reused_topk_ids,
        reused_expert_map,
        num_experts,
        block_size,
        *reused_outputs,
    )


@pytest.mark.parametrize(
    "metadata_change",
    (
        "topk_shape",
        "topk_strides",
        "topk_dtype",
        "sorted_token_ids_size",
        "expert_map_shape",
        "expert_map_presence",
    ),
)
def test_moe_align_block_size_descriptor_rejects_changed_metadata(
    metadata_change, device, implementation_index
):
    if device != "cuda":
        pytest.skip("`moe_align_block_size` requires the NVIDIA backend")

    assertion_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            _DESCRIPTOR_REUSE_SCRIPT,
            "assertions_enabled_probe",
        ],
        capture_output=True,
        text=True,
    )

    if assertion_probe.returncode == 0:
        pytest.skip("descriptor validation requires an assertions-enabled build")

    result = subprocess.run(
        [sys.executable, "-c", _DESCRIPTOR_REUSE_SCRIPT, metadata_change],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "tensor metadata differs from its descriptor" in result.stderr


def test_moe_align_block_size_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    topk_ids = torch.tensor(
        ((0, 1), (2, 3), (0, 3), (2, 1)), dtype=torch.int32, device=device
    )
    num_experts = 4
    block_size = 4
    outputs = _make_outputs(topk_ids, num_experts, block_size)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    infini.ops.moe_align_block_size(
        topk_ids,
        None,
        num_experts,
        block_size,
        *outputs,
        stream=stream.cuda_stream,
        implementation_index=implementation_index,
    )

    stream.synchronize()
    _assert_matches_reference(topk_ids, None, num_experts, block_size, *outputs)


def test_moe_align_block_size_device_guard():
    if not infini.ops.MoeAlignBlockSize.active_implementation_indices("nvidia"):
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
        block_size = 4
        outputs = _make_outputs(topk_ids, num_experts, block_size)
        stream = torch.cuda.Stream(device=target_device)
        stream.wait_stream(torch.cuda.current_stream(target_device))

        infini.ops.moe_align_block_size(
            topk_ids,
            None,
            num_experts,
            block_size,
            *outputs,
            stream=stream.cuda_stream,
        )

        assert torch.cuda.current_device() == 0
        stream.synchronize()
        _assert_matches_reference(topk_ids, None, num_experts, block_size, *outputs)
    finally:
        torch.cuda.set_device(original_device)


def _make_outputs(topk_ids, num_experts, block_size):
    numel = topk_ids.numel()
    max_num_tokens_padded = numel + num_experts * (block_size - 1)

    if numel < num_experts:
        max_num_tokens_padded = min(numel * block_size, max_num_tokens_padded)

    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,), -2, dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.full(
        (max_num_blocks,), -2, dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.full(
        (1,), -2, dtype=torch.int32, device=topk_ids.device
    )

    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _moe_align_block_size(
    topk_ids,
    expert_map,
    num_experts,
    block_size,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
    *,
    implementation_index,
):
    infini.ops.moe_align_block_size(
        topk_ids,
        expert_map,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        stream=get_stream(topk_ids.device),
        implementation_index=implementation_index,
    )


def _assert_matches_reference(
    topk_ids,
    expert_map,
    num_experts,
    block_size,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
):
    flattened_experts = topk_ids.flatten().to(torch.int64)

    if expert_map is not None:
        mapped_experts = torch.full_like(flattened_experts, -1)
        valid = (flattened_experts >= 0) & (flattened_experts < num_experts)
        mapped_experts[valid] = expert_map[flattened_experts[valid]].to(torch.int64)
    else:
        mapped_experts = flattened_experts

    expected_sorted_token_ids = []
    expected_blocks = []

    for expert_id in range(num_experts):
        token_ids = torch.nonzero(mapped_experts == expert_id).flatten().tolist()

        if not token_ids:
            continue

        num_blocks = (len(token_ids) + block_size - 1) // block_size
        expected_sorted_token_ids.extend(token_ids)
        expected_sorted_token_ids.extend(
            [topk_ids.numel()] * (num_blocks * block_size - len(token_ids))
        )
        expected_blocks.extend([expert_id] * num_blocks)

    expected_num_tokens = len(expected_blocks) * block_size
    assert num_tokens_post_pad.item() == expected_num_tokens
    assert expert_ids[: len(expected_blocks)].tolist() == expected_blocks
    assert torch.all(expert_ids[len(expected_blocks) :] == -1)
    assert sorted_token_ids[:expected_num_tokens].tolist() == expected_sorted_token_ids
    assert torch.all(sorted_token_ids[expected_num_tokens:] == topk_ids.numel())


_DESCRIPTOR_REUSE_SCRIPT = textwrap.dedent(
    r"""
    import sys

    import infini.ops
    import torch


    metadata_change = sys.argv[1]
    num_experts = 4
    block_size = 2
    topk_ids = torch.tensor(((0, 1), (2, 3)), dtype=torch.int32, device="cuda")
    expert_map = torch.arange(num_experts, dtype=torch.int32, device="cuda")
    sorted_token_ids = torch.empty(8, dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(4, dtype=torch.int32, device="cuda")
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device="cuda")
    operator = infini.ops.MoeAlignBlockSize(
        topk_ids,
        expert_map,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    if metadata_change == "assertions_enabled_probe":
        infini.ops.MoeAlignBlockSize(
            topk_ids.reshape(4),
            expert_map,
            num_experts,
            block_size,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
        )
    elif metadata_change == "topk_shape":
        topk_ids = topk_ids.reshape(1, 4)
    elif metadata_change == "topk_strides":
        topk_ids = topk_ids.T
    elif metadata_change == "topk_dtype":
        topk_ids = topk_ids.to(torch.int64)
    elif metadata_change == "sorted_token_ids_size":
        sorted_token_ids = torch.empty(9, dtype=torch.int32, device="cuda")
    elif metadata_change == "expert_map_shape":
        expert_map = expert_map[:3]

    if metadata_change == "expert_map_presence":
        operator(
            topk_ids,
            None,
            num_experts,
            block_size,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
        )
    else:
        operator(
            topk_ids,
            expert_map,
            num_experts,
            block_size,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
        )
    torch.cuda.synchronize()
    """
)
