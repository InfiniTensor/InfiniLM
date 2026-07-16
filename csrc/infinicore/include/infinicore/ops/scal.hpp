#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Scal, const Tensor &, Tensor);

void scal_(const Tensor &alpha, Tensor x);

} // namespace infinicore::op
