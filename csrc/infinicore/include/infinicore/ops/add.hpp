#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Add, Tensor, const Tensor &, const Tensor &);

Tensor add(const Tensor &a, const Tensor &b);
void add_(Tensor c, const Tensor &a, const Tensor &b);

} // namespace infinicore::op
