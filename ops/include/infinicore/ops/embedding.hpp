#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Embedding, Tensor, const Tensor &, const Tensor &);

Tensor embedding(const Tensor &input, const Tensor &weight);
void embedding_(Tensor out, const Tensor &input, const Tensor &weight);
} // namespace infinicore::op
