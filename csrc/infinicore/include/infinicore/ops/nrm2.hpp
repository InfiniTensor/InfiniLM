#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Nrm2, const Tensor &, Tensor);

Tensor nrm2(const Tensor &x);
void nrm2_(const Tensor &x, Tensor result);

} // namespace infinicore::op
