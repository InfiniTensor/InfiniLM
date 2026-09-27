#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Asum, const Tensor &, Tensor);

Tensor asum(const Tensor &x);
void asum_(const Tensor &x, Tensor result);

} // namespace infinicore::op
