#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(CausalSoftmax, Tensor, const Tensor &);

Tensor causal_softmax(const Tensor &input);
void causal_softmax_(Tensor output, const Tensor &input);

} // namespace infinicore::op
