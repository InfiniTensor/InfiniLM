#pragma once

#include "infinicore/tensor.hpp"

#include <optional>

namespace infinilm::vllm_fused_moe_dispatch {

/// Returns whether the fused MoE bridge is usable (vendored ``infinicore.vendor.vllm_fused_moe``).
/// Cached: once unavailable (e.g. import/Triton failure), stays disabled for the process.
bool fused_experts_ic_available();

/// Preferred path (InfiniLM built with ``xmake f --aten=y``): ``to_aten`` views +
/// ``c10::Dispatcher`` call to ``infinilm::outplace_fused_experts`` (SILU, defaults matching
/// ``fused_experts_ic``). Set ``INFINILM_VLLM_FUSED_DISPATCH=legacy`` to force the Python
/// ``infinicore.vllm_fused_moe_bridge.fused_experts_ic`` route. Returns nullopt on failure.
std::optional<infinicore::Tensor> try_fused_experts_ic(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &w1_stacked,
    const infinicore::Tensor &w2_stacked,
    const infinicore::Tensor &topk_weights,
    const infinicore::Tensor &topk_ids);

} // namespace infinilm::vllm_fused_moe_dispatch
