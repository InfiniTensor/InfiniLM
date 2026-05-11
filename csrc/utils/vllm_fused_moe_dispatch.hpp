#pragma once

#include "infinicore/tensor.hpp"

#include <optional>

namespace infinilm::vllm_fused_moe_dispatch {

/// Returns whether the vLLM fused experts bridge is usable in this interpreter.
/// This is cached: once it is detected as unavailable (e.g. vLLM not installed),
/// it stays disabled for the rest of the process to avoid per-token Python errors.
bool fused_experts_ic_available();

/// Preferred path (InfiniLM built with ``xmake f --aten=y``): ``to_aten`` views +
/// ``c10::Dispatcher`` call to ``vllm::outplace_fused_experts`` (SILU, defaults matching
/// ``fused_experts_ic``). Set ``INFINILM_VLLM_FUSED_DISPATCH=legacy`` to force the Python
/// ``infinicore.vllm_fused_moe_bridge.fused_experts_ic`` route. Returns nullopt on failure.
/// Throughput vs legacy is not guaranteed (same underlying vLLM op; boxed dispatch and GIL
/// still apply). Prefer ``legacy`` if profiling shows it faster on your stack.
std::optional<infinicore::Tensor> try_fused_experts_ic(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &w1_stacked,
    const infinicore::Tensor &w2_stacked,
    const infinicore::Tensor &topk_weights,
    const infinicore::Tensor &topk_ids);

} // namespace infinilm::vllm_fused_moe_dispatch
