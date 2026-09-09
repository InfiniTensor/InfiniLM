#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dist, Tensor, const Tensor &, const Tensor &, double);

Tensor dist(const Tensor &x1, const Tensor &x2, double p = 2.0);
void dist_(Tensor y, const Tensor &x1, const Tensor &x2, double p = 2.0);

} // namespace infinicore::op
