#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(MoeTopkSigmoid,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const bool);

std::tuple<Tensor, Tensor> moe_topk_sigmoid(
    const Tensor &gating_output,
    size_t topk,
    bool renormalize = false,
    const Tensor &correction_bias = Tensor());

void moe_topk_sigmoid_(
    Tensor topk_weights,
    Tensor topk_indices,
    const Tensor &gating_output,
    const Tensor &correction_bias = Tensor(),
    bool renormalize = false);

} // namespace infinicore::op
