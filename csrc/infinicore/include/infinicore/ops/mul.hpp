#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Mul, Tensor, const Tensor &, const Tensor &);

Tensor mul(const Tensor &a, const Tensor &b);
void mul_(Tensor c, const Tensor &a, const Tensor &b);

} // namespace infinicore::op
