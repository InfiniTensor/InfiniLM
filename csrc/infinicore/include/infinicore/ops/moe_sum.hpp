#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(MoeSum, Tensor, const Tensor &);

Tensor moe_sum(const Tensor &input);
void moe_sum_(Tensor output, const Tensor &input);

} // namespace infinicore::op
