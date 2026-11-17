#pragma once

#if defined(ENABLE_ATEN) && defined(ENABLE_QY_API)
#include <ATen/ATen.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace infinicore::adaptor::denglin_fused_moe {

/** True when vllm._dl_C.invoke_fused_moe_opt is loadable (same backend as dl-vllm). */
bool available();

/**
 * Thin wrapper around Denglin/vLLM ``invoke_fused_moe_opt`` (torch.ops._dl_C).
 * Matches vLLM ``dl_fused_moe.dl_invoke_fused_moe`` calling convention.
 */
void invoke_fused_moe_opt(
    at::Tensor &a,
    at::Tensor &b,
    at::Tensor &c,
    const std::optional<at::Tensor> &b_bias,
    const std::optional<at::Tensor> &b_scale,
    const std::optional<at::Tensor> &b_zp,
    at::Tensor &topk_weights,
    at::Tensor &topk_ids,
    at::Tensor &sorted_token_ids,
    at::Tensor &expert_ids,
    at::Tensor &num_tokens_post_padded,
    bool mul_routed_weight,
    int64_t top_k,
    int64_t block_size_m,
    int64_t block_size_n,
    int64_t block_size_k,
    bool use_fp8_w8a8,
    bool use_int8_w8a16,
    bool use_int4_w4a16,
    bool use_mxfp4_w4a16,
    const std::vector<int64_t> &block_shape,
    int64_t m);

} // namespace infinicore::adaptor::denglin_fused_moe
#endif
