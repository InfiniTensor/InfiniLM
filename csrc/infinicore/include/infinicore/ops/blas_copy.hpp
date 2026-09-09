#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(BlasCopy, const Tensor &, Tensor);

void blas_copy_(const Tensor &x, Tensor y);

} // namespace infinicore::op
