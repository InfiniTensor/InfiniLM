#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SiluAndMul, Tensor, const Tensor &);

Tensor silu_and_mul(const Tensor &x);
void silu_and_mul_(Tensor out, const Tensor &x);

} // namespace infinicore::op
