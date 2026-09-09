#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(BlasAmin, const Tensor &, Tensor);

Tensor blas_amin(const Tensor &x);
void blas_amin_(const Tensor &x, Tensor result);

} // namespace infinicore::op
