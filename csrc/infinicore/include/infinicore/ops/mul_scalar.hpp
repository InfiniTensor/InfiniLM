#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

// MiniCPM5 MoE routing-weight scale; capture-safe under full_and_piecewise.
INFINICORE_GRAPH_OP_CLASS(MulScalar, Tensor, const Tensor &, double);

Tensor mul_scalar(const Tensor &a, double alpha);
void mul_scalar_(Tensor c, const Tensor &a, double alpha);

} // namespace infinicore::op
