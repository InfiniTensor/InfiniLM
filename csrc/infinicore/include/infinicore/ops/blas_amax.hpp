#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(BlasAmax, const Tensor &, Tensor);

Tensor blas_amax(const Tensor &x);
void blas_amax_(const Tensor &x, Tensor result);

} // namespace infinicore::op
