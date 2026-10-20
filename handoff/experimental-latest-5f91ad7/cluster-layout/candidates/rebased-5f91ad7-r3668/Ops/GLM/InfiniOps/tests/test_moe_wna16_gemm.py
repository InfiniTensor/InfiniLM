import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "MoeWna16Gemm"):
    pytest.skip(
        "`MoeWna16Gemm` is not available on this platform",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "bit, group_size, k, block_size_k",
    (
        (4, 32, 32, 32),
        (4, 16, 64, 32),
        (4, 8, 32, 32),
        (4, 8, 64, 64),
        (8, 32, 32, 32),
        (8, 16, 32, 32),
        (8, 8, 32, 32),
        (8, 4, 32, 32),
    ),
)
@pytest.mark.parametrize("has_zero_point", (False, True))
@pytest.mark.parametrize("multiply_topk_weight", (False, True))
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 5e-2, 5e-2),
    ),
)
def test_moe_wna16_gemm(
    bit,
    group_size,
    k,
    block_size_k,
    has_zero_point,
    multiply_topk_weight,
    dtype,
    device,
    implementation_index,
    rtol,
    atol,
):
    if device != "cuda":
        pytest.skip("`moe_wna16_gemm` requires the NVIDIA backend")

    tensors = _make_case(
        bit,
        has_zero_point,
        multiply_topk_weight,
        dtype,
        device,
        group_size=group_size,
        k=k,
    )
    (
        input,
        qweight,
        scales,
        qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        output,
        quantized_weight,
        zero_points,
    ) = tensors

    result = infini.ops.moe_wna16_gemm(
        input,
        qweight,
        scales,
        qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        2,
        2,
        16,
        block_size_k,
        bit,
        output,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    assert result is None
    expected = _reference(
        input,
        quantized_weight,
        scales,
        zero_points,
        bit,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        block_size_m=2,
    )
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


def test_moe_wna16_gemm_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    tensors = _make_case(4, True, True, torch.float16, device)
    (
        input,
        qweight,
        scales,
        qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        output,
        quantized_weight,
        zero_points,
    ) = tensors
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    infini.ops.moe_wna16_gemm(
        input,
        qweight,
        scales,
        qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        2,
        2,
        16,
        32,
        4,
        output,
        stream=stream.cuda_stream,
        implementation_index=implementation_index,
    )

    stream.synchronize()
    expected = _reference(
        input,
        quantized_weight,
        scales,
        zero_points,
        4,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        block_size_m=2,
    )
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_moe_wna16_gemm_skips_padding_and_unmapped_expert(device, implementation_index):
    if device != "cuda":
        pytest.skip("`moe_wna16_gemm` requires the NVIDIA backend")

    tensors = _make_case(
        4,
        True,
        True,
        torch.float16,
        device,
        num_tokens=3,
        n=24,
        sorted_token_ids_values=(0, 1, 2, 6, 4, 5, 4, 5),
        expert_ids_values=(0, 1, -1, 0),
        num_tokens_post_pad_value=6,
    )
    (
        input,
        qweight,
        scales,
        qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        output,
        quantized_weight,
        zero_points,
    ) = tensors

    infini.ops.moe_wna16_gemm(
        input,
        qweight,
        scales,
        qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        2,
        2,
        16,
        32,
        4,
        output,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    expected = _reference(
        input,
        quantized_weight,
        scales,
        zero_points,
        4,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        block_size_m=2,
    )
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
    assert torch.count_nonzero(output.flatten(0, 1)[3:]).item() == 0


@pytest.mark.parametrize("device, implementation_index", (("cuda", 16),))
def test_moe_wna16_gemm_linked_vllm(device, implementation_index):
    if (
        implementation_index
        not in infini.ops.MoeWna16Gemm.active_implementation_indices(device)
    ):
        pytest.skip("vLLM linked implementation is not active")

    tensors = _make_case(4, True, True, torch.float16, device)
    (
        input,
        qweight,
        scales,
        qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        output,
        quantized_weight,
        zero_points,
    ) = tensors

    result = infini.ops.moe_wna16_gemm(
        input,
        qweight,
        scales,
        qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        2,
        2,
        16,
        32,
        4,
        output,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    assert result is None
    expected = _reference(
        input,
        quantized_weight,
        scales,
        zero_points,
        4,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        block_size_m=2,
    )
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def _make_case(
    bit,
    has_zero_point,
    multiply_topk_weight,
    dtype,
    device,
    *,
    group_size=16,
    k=32,
    num_tokens=2,
    n=16,
    sorted_token_ids_values=(0, 2, 1, 3),
    expert_ids_values=(0, 1),
    num_tokens_post_pad_value=4,
):
    torch.manual_seed(0)
    top_k = 2
    num_experts = 2
    num_groups = k // group_size
    maximum = 1 << bit

    input = torch.randn((num_tokens, k), dtype=dtype, device=device)
    quantized_weight = torch.randint(
        0,
        maximum,
        (num_experts, n, k),
        dtype=torch.int64,
        device=device,
    )
    qweight = _pack_last_dimension(quantized_weight, bit)
    scales = (torch.rand((num_experts, n, num_groups), device=device) * 0.05 + 0.01).to(
        dtype
    )

    zero_points = None
    qzeros = None
    if has_zero_point:
        zero_points = torch.randint(
            maximum // 4,
            3 * maximum // 4,
            (num_experts, n, num_groups),
            dtype=torch.int64,
            device=device,
        )
        qzeros = _pack_n_dimension(zero_points, bit)

    topk_weights = None
    if multiply_topk_weight:
        topk_weights = torch.rand(
            (num_tokens, top_k), dtype=torch.float32, device=device
        )

    sorted_token_ids = torch.tensor(
        sorted_token_ids_values, dtype=torch.int32, device=device
    )
    expert_ids = torch.tensor(expert_ids_values, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.tensor(
        (num_tokens_post_pad_value,), dtype=torch.int32, device=device
    )
    output = torch.full((num_tokens, top_k, n), torch.nan, dtype=dtype, device=device)

    return (
        input,
        qweight,
        scales,
        qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        output,
        quantized_weight,
        zero_points,
    )


def _pack_last_dimension(values, bit):
    if bit == 8:
        return values.to(torch.uint8).contiguous()

    low = values[..., 0::2]
    high = values[..., 1::2]
    return (low | (high << 4)).to(torch.uint8).contiguous()


def _pack_n_dimension(values, bit):
    if bit == 8:
        return values.to(torch.uint8).contiguous()

    low = values[:, 0::2, :]
    high = values[:, 1::2, :]
    return (low | (high << 4)).to(torch.uint8).contiguous()


def _reference(
    input,
    quantized_weight,
    scales,
    zero_points,
    bit,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
    *,
    block_size_m,
):
    group_size = input.size(1) // scales.size(2)
    expanded_scales = scales.float().repeat_interleave(group_size, dim=2)
    if zero_points is None:
        midpoint = 1 << (bit - 1)
        expanded_zero_points = midpoint
    else:
        expanded_zero_points = zero_points.float().repeat_interleave(group_size, dim=2)
    weights = (quantized_weight.float() - expanded_zero_points) * expanded_scales

    expected = torch.zeros(
        (
            input.size(0),
            topk_weights.size(1) if topk_weights is not None else 2,
            quantized_weight.size(1),
        ),
        dtype=torch.float32,
        device=input.device,
    )
    flat_expected = expected.flatten(0, 1)
    top_k = expected.size(1)
    cutoff = num_tokens_post_pad.item()
    for block, expert in enumerate(expert_ids.tolist()):
        if block * block_size_m >= cutoff:
            break
        if expert == -1:
            continue
        for route in sorted_token_ids[
            block * block_size_m : (block + 1) * block_size_m
        ].tolist():
            token = route // top_k
            if token >= input.size(0):
                break
            if input.dtype == torch.bfloat16 and bit == 8:
                value = _bfloat16_w8_reference(
                    input[token],
                    quantized_weight[expert],
                    scales[expert],
                    None if zero_points is None else zero_points[expert],
                )
            else:
                value = input[token].float() @ weights[expert].transpose(0, 1)
            if topk_weights is not None:
                value *= topk_weights.flatten()[route]
            flat_expected[route] += value

    return expected.to(input.dtype)


def _bfloat16_w8_reference(activation, quantized_weight, scales, zero_points):
    group_size = activation.numel() // scales.size(1)
    result = torch.zeros(
        quantized_weight.size(0), dtype=torch.float32, device=activation.device
    )

    for packed_k in range(0, activation.numel(), 4):
        scale = scales[:, packed_k // group_size]
        if zero_points is None:
            zero_point = torch.full_like(scale, 128)
        else:
            zero_point = zero_points[:, packed_k // group_size].to(torch.bfloat16)

        quantized = quantized_weight[:, packed_k : packed_k + 4].to(torch.bfloat16)
        weight = (
            (quantized - zero_point[:, None]).to(torch.bfloat16) * scale[:, None]
        ).to(torch.bfloat16)

        lane0 = (weight[:, 0] * activation[packed_k]).to(torch.bfloat16)
        lane0 = (
            weight[:, 1].float() * activation[packed_k + 1].float() + lane0.float()
        ).to(torch.bfloat16)
        lane1 = (weight[:, 2] * activation[packed_k + 2]).to(torch.bfloat16)
        lane1 = (
            weight[:, 3].float() * activation[packed_k + 3].float() + lane1.float()
        ).to(torch.bfloat16)
        result += lane0.float() + lane1.float()

    return result
