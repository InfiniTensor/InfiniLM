#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(BlasDot, const Tensor &, const Tensor &, Tensor);

Tensor blas_dot(const Tensor &x, const Tensor &y);
void blas_dot_(const Tensor &x, const Tensor &y, Tensor result);

} // namespace infinicore::op
