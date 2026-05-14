#pragma once

#include "infinicore/tensor.hpp"

#include <optional>

namespace infinilm::vllm_fused_moe_dispatch {

/// ``INFINILM_MOE_FUSED_STACK=upstream`` (case-insensitive). When unset, not upstream.
/// Keep in sync with ``infinicore.moe_fused_stack.resolve_moe_fused_stack``.
bool moe_fused_stack_upstream_env();

/// ``INFINILM_MOE_FUSED_STACK=vendor_router_cpu`` (case-insensitive): vendor fused experts with CPU routing path.
bool moe_fused_stack_vendor_router_cpu_env();

/// Returns whether the fused MoE bridge is usable (vendored ``infinicore.vendor.vllm_fused_moe``).
/// Cached: once unavailable (e.g. import/Triton failure), stays disabled for the process.
bool fused_experts_ic_available();

struct GroupedSigmoidTopkIcResult {
    infinicore::Tensor topk_weights;
    infinicore::Tensor topk_ids;
};

/// Device grouped sigmoid router (``torch.ops.infinilm.minicpm5_grouped_sigmoid_topk`` via Python bridge).
/// Returns nullopt on failure (e.g. import error). Caller should fall back to CPU routing if needed.
std::optional<GroupedSigmoidTopkIcResult> try_grouped_sigmoid_topk_ic(
    const infinicore::Tensor &router_logits_f32,
    const infinicore::Tensor &e_score_correction_bias,
    size_t top_k,
    bool norm_topk_prob,
    float routed_scaling_factor,
    size_t n_group,
    size_t topk_group);

/// Preferred path (InfiniLM built with ``xmake f --aten=y``): ``to_aten`` views +
/// ``c10::Dispatcher`` call to ``infinilm::outplace_fused_experts`` (SILU, defaults matching
/// ``fused_experts_ic``). Skipped when ``moe_fused_stack_upstream_env()`` is true (upstream uses
/// ``torch.ops.vllm.*`` via Python). Set ``INFINILM_VLLM_FUSED_DISPATCH=legacy`` to force the Python
/// ``infinicore.vllm_fused_moe_bridge.fused_experts_ic`` route. Returns nullopt on failure.
std::optional<infinicore::Tensor> try_fused_experts_ic(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &w1_stacked,
    const infinicore::Tensor &w2_stacked,
    const infinicore::Tensor &topk_weights,
    const infinicore::Tensor &topk_ids);

} // namespace infinilm::vllm_fused_moe_dispatch
