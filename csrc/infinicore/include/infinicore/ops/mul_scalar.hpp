#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

// Used in MiniCPM5 MoE routing-weight scale; keep eager with MoE segments.
INFINICORE_GRAPH_OP_CLASS_CAPTURE_UNSAFE(MulScalar, Tensor, const Tensor &, double);

Tensor mul_scalar(const Tensor &a, double alpha);
void mul_scalar_(Tensor c, const Tensor &a, double alpha);

} // namespace infinicore::op
