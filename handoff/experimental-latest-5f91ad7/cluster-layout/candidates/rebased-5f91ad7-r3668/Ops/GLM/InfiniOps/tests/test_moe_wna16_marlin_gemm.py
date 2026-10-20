import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "MoeWna16MarlinGemm"):
    pytest.skip(
        "`MoeWna16MarlinGemm` is not available on this platform",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "dtype, b_type_id, num_bits, rtol, atol",
    (
        (torch.float16, 1125899907892224, 4, 2e-2, 2e-2),
        (torch.bfloat16, 1125899907892224, 4, 5e-2, 5e-2),
        (torch.float16, 1125899923621888, 8, 2e-2, 2e-2),
    ),
)
def test_moe_wna16_marlin_gemm(
    dtype,
    b_type_id,
    num_bits,
    rtol,
    atol,
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("`moe_wna16_marlin_gemm` requires the NVIDIA backend")

    provider_case = _make_case(device, dtype, b_type_id, num_bits)
    case = _make_case(device, dtype, b_type_id, num_bits)
    expected = provider_case["out"]
    provider_result = _call_provider(provider_case, expected)

    assert provider_result.data_ptr() == expected.data_ptr()
    result = _call_infini(case, implementation_index, get_stream(case["a"].device))

    assert result is None
    torch.testing.assert_close(case["out"], expected, rtol=rtol, atol=atol)


def test_moe_wna16_marlin_gemm_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    provider_case = _make_case(device, torch.float16, 1125899907892224, 4)
    case = _make_case(device, torch.float16, 1125899907892224, 4)
    expected = provider_case["out"]
    _call_provider(provider_case, expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    _call_infini(case, implementation_index, stream.cuda_stream)

    stream.synchronize()
    torch.testing.assert_close(case["out"], expected, rtol=2e-2, atol=2e-2)


def test_moe_wna16_marlin_gemm_optional_act_order(device, implementation_index):
    if device != "cuda":
        pytest.skip("activation-order metadata requires the NVIDIA backend")

    provider_case = _make_case(
        device, torch.float16, 1125899907892224, 4, has_act_order=True
    )
    case = _make_case(device, torch.float16, 1125899907892224, 4, has_act_order=True)
    expected = provider_case["out"]
    _call_provider(provider_case, expected)

    _call_infini(case, implementation_index, get_stream(case["a"].device))

    torch.testing.assert_close(case["out"], expected, rtol=2e-2, atol=2e-2)


def test_moe_wna16_marlin_gemm_zero_points(device, implementation_index):
    if device != "cuda":
        pytest.skip("zero-point metadata requires the NVIDIA backend")

    provider_case = _make_case(
        device, torch.float16, 1125899906843648, 4, has_zero_points=True
    )
    case = _make_case(device, torch.float16, 1125899906843648, 4, has_zero_points=True)
    expected = provider_case["out"]
    provider_result = _call_provider(provider_case, expected)

    assert provider_result.data_ptr() == expected.data_ptr()
    _call_infini(case, implementation_index, get_stream(case["a"].device))

    torch.testing.assert_close(case["out"], expected, rtol=2e-2, atol=2e-2)


def test_moe_wna16_marlin_gemm_nonzero_bias_and_explicit_tuning(
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("bias and tuning require the NVIDIA backend")

    case_kwargs = {
        "has_bias": True,
        "mul_topk_weights": False,
        "size_n": 256,
        "size_k": 256,
        "thread_config": (128, 128, 1),
        "top_k": 1,
        "topk_weight_dtype": torch.float16,
        "use_atomic_add": False,
        "zero_scales": True,
    }
    provider_case = _make_case(
        device, torch.float16, 1125899907892224, 4, **case_kwargs
    )
    case = _make_case(device, torch.float16, 1125899907892224, 4, **case_kwargs)
    expected = provider_case["out"]
    _call_provider(provider_case, expected)

    assert torch.count_nonzero(expected).item() > 0
    _call_infini(case, implementation_index, get_stream(case["a"].device))

    torch.testing.assert_close(case["out"], expected, rtol=2e-2, atol=2e-2)


def test_moe_wna16_marlin_gemm_int8_activation_scales(
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("int8 activation scales require the NVIDIA backend")

    case_kwargs = {
        "activation_dtype": torch.int8,
        "mul_topk_weights": False,
        "size_n": 1024,
        "size_k": 1024,
        "top_k": 1,
        "topk_weight_dtype": torch.float16,
        "use_atomic_add": False,
    }
    provider_case = _make_case(
        device, torch.float16, 1125899907892224, 4, **case_kwargs
    )
    case = _make_case(device, torch.float16, 1125899907892224, 4, **case_kwargs)
    expected = provider_case["out"]
    _call_provider(provider_case, expected)

    assert torch.count_nonzero(expected).item() > 0
    _call_infini(case, implementation_index, get_stream(case["a"].device))

    torch.testing.assert_close(case["out"], expected, rtol=2e-2, atol=2e-2)


def test_moe_wna16_marlin_gemm_accepts_flat_topk_weights(
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("top-k weights require the NVIDIA backend")

    provider_case = _make_case(device, torch.float16, 1125899907892224, 4)
    case = _make_case(device, torch.float16, 1125899907892224, 4)
    provider_case["topk_weights"] = provider_case["topk_weights"].reshape(-1)
    case["topk_weights"] = case["topk_weights"].reshape(-1)
    expected = provider_case["out"]
    _call_provider(provider_case, expected)

    _call_infini(case, implementation_index, get_stream(case["a"].device))

    torch.testing.assert_close(case["out"], expected, rtol=2e-2, atol=2e-2)


def _make_case(
    device,
    dtype,
    b_type_id,
    num_bits,
    *,
    activation_dtype=None,
    has_act_order=False,
    has_bias=False,
    has_zero_points=False,
    mul_topk_weights=True,
    size_n=128,
    size_k=256,
    thread_config=(-1, -1, -1),
    top_k=2,
    topk_weight_dtype=torch.float32,
    use_atomic_add=None,
    zero_scales=False,
):
    torch.manual_seed(0)
    size_m = 1
    num_experts = 4
    moe_block_size = 16
    route_count = size_m * top_k
    padding = route_count
    sorted_token_ids = torch.full(
        (route_count + num_experts * (moe_block_size - 1),),
        padding,
        dtype=torch.int32,
        device=device,
    )
    sorted_token_ids[torch.arange(route_count, device=device) * moe_block_size] = (
        torch.arange(route_count, dtype=torch.int32, device=device)
    )
    expert_ids = torch.zeros((num_experts,), dtype=torch.int32, device=device)
    expert_ids[:route_count] = torch.arange(
        route_count, dtype=torch.int32, device=device
    )
    pack_factor = 32 // num_bits
    num_groups = 8 if has_act_order else 1
    scale = 0.0 if has_act_order or zero_scales else 0.02
    g_idx_or_none = None
    perm_or_none = None
    thread_k, thread_n, blocks_per_sm = thread_config
    activation_dtype = activation_dtype or dtype
    a_scales = None
    if activation_dtype == torch.int8:
        a = torch.ones((size_m, size_k), dtype=activation_dtype, device=device)
        a_scales = torch.full((size_m, 1), 0.01, dtype=torch.float32, device=device)
    else:
        a = torch.randn((size_m, size_k), dtype=activation_dtype, device=device)

    b_bias_or_none = None
    if has_bias:
        b_bias_or_none = torch.full(
            (num_experts, size_n), 0.125, dtype=dtype, device=device
        )

    if has_act_order:
        g_idx_or_none = (
            torch.arange(num_groups, dtype=torch.int32, device=device)
            .repeat_interleave(size_k // num_groups)
            .repeat(num_experts, 1)
        )
        perm_or_none = torch.arange(size_k, dtype=torch.int32, device=device).repeat(
            num_experts, 1
        )

    b_zeros_or_none = None
    if has_zero_points:
        b_zeros_or_none = torch.zeros(
            (num_experts, num_groups, size_n // pack_factor),
            dtype=torch.int32,
            device=device,
        )

    if use_atomic_add is None:
        use_atomic_add = dtype == torch.float16

    return {
        "a": a,
        "b_q_weight": torch.zeros(
            (num_experts, size_k // 16, size_n * 16 // pack_factor),
            dtype=torch.int32,
            device=device,
        ),
        "b_bias_or_none": b_bias_or_none,
        "b_scales": torch.full(
            (num_experts, num_groups, size_n), scale, dtype=dtype, device=device
        ),
        "a_scales": a_scales,
        "global_scale": None,
        "b_zeros_or_none": b_zeros_or_none,
        "g_idx_or_none": g_idx_or_none,
        "perm_or_none": perm_or_none,
        "workspace": torch.zeros(432, dtype=torch.int32, device=device),
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "num_tokens_past_padded": torch.tensor(
            (route_count * moe_block_size,), dtype=torch.int32, device=device
        ),
        "topk_weights": torch.full(
            (size_m, top_k), 1.0 / top_k, dtype=topk_weight_dtype, device=device
        ),
        "moe_block_size": moe_block_size,
        "top_k": top_k,
        "mul_topk_weights": mul_topk_weights,
        "b_type_id": b_type_id,
        "size_m": size_m,
        "size_n": size_n,
        "size_k": size_k,
        "is_full_k": True,
        "use_atomic_add": use_atomic_add,
        "use_fp32_reduce": True,
        "is_zp_float": False,
        "thread_k": thread_k,
        "thread_n": thread_n,
        "blocks_per_sm": blocks_per_sm,
        "out": torch.zeros((route_count, size_n), dtype=dtype, device=device),
    }


def _call_provider(case, out):
    return torch.ops._moe_C.moe_wna16_marlin_gemm(
        case["a"],
        out,
        case["b_q_weight"],
        case["b_bias_or_none"],
        case["b_scales"],
        case["a_scales"],
        case["global_scale"],
        case["b_zeros_or_none"],
        case["g_idx_or_none"],
        case["perm_or_none"],
        case["workspace"],
        case["sorted_token_ids"],
        case["expert_ids"],
        case["num_tokens_past_padded"],
        case["topk_weights"],
        case["moe_block_size"],
        case["top_k"],
        case["mul_topk_weights"],
        case["b_type_id"],
        case["size_m"],
        case["size_n"],
        case["size_k"],
        case["is_full_k"],
        case["use_atomic_add"],
        case["use_fp32_reduce"],
        case["is_zp_float"],
        case["thread_k"],
        case["thread_n"],
        case["blocks_per_sm"],
    )


def _call_infini(case, implementation_index, stream):
    return infini.ops.moe_wna16_marlin_gemm(
        case["a"],
        case["b_q_weight"],
        case["b_bias_or_none"],
        case["b_scales"],
        case["a_scales"],
        case["global_scale"],
        case["b_zeros_or_none"],
        case["g_idx_or_none"],
        case["perm_or_none"],
        case["workspace"],
        case["sorted_token_ids"],
        case["expert_ids"],
        case["num_tokens_past_padded"],
        case["topk_weights"],
        case["moe_block_size"],
        case["top_k"],
        case["mul_topk_weights"],
        case["b_type_id"],
        case["size_m"],
        case["size_n"],
        case["size_k"],
        case["is_full_k"],
        case["use_atomic_add"],
        case["use_fp32_reduce"],
        case["is_zp_float"],
        case["thread_k"],
        case["thread_n"],
        case["blocks_per_sm"],
        case["out"],
        stream=stream,
        implementation_index=implementation_index,
    )
