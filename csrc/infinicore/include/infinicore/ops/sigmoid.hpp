#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Sigmoid, Tensor, const Tensor &);

Tensor sigmoid(const Tensor &input);
void sigmoid_(Tensor output, const Tensor &input);

} // namespace infinicore::op
