#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(MoeFusedGate,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const size_t,
                          const size_t,
                          const size_t,
                          const float,
                          const bool);

std::tuple<Tensor, Tensor> moe_fused_gate(
    const Tensor &input,
    const Tensor &bias,
    size_t topk,
    size_t num_expert_group,
    size_t topk_group,
    size_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output);

void moe_fused_gate_(
    Tensor topk_weights,
    Tensor topk_indices,
    const Tensor &input,
    const Tensor &bias,
    size_t num_expert_group,
    size_t topk_group,
    size_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output);

} // namespace infinicore::op
