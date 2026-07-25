#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Digamma, Tensor, const Tensor &);

Tensor digamma(const Tensor &x);
void digamma_(Tensor y, const Tensor &x);

} // namespace infinicore::op
