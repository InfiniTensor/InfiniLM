#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Axpy, const Tensor &, const Tensor &, Tensor);

void axpy_(const Tensor &alpha, const Tensor &x, Tensor y);

} // namespace infinicore::op
