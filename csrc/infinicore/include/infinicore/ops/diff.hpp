#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Diff, Tensor, const Tensor &, int, int);

Tensor diff(const Tensor &x, int n = 1, int dim = -1);
void diff_(Tensor y, const Tensor &x, int n = 1, int dim = -1);

} // namespace infinicore::op
