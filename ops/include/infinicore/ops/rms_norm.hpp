#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(RMSNorm, Tensor, const Tensor &, const Tensor &, float);

Tensor rms_norm(const Tensor &x, const Tensor &weight, float epsilon = 1e-5f);
void rms_norm_(Tensor y, const Tensor &x, const Tensor &weight, float epsilon = 1e-5f);

} // namespace infinicore::op
