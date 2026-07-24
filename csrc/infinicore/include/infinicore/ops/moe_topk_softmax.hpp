#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(MoeTopkSoftmax,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const bool,
                          const float);

std::tuple<Tensor, Tensor> moe_topk_softmax(
    const Tensor &gating_output,
    size_t topk,
    bool renormalize = false,
    float moe_softcapping = 0.0f,
    const Tensor &correction_bias = Tensor());

void moe_topk_softmax_(
    Tensor topk_weights,
    Tensor topk_indices,
    const Tensor &gating_output,
    const Tensor &correction_bias = Tensor(),
    bool renormalize = false,
    float moe_softcapping = 0.0f);

} // namespace infinicore::op
