#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Rot, Tensor, Tensor, const Tensor &, const Tensor &);

void rot_(Tensor x, Tensor y, const Tensor &c, const Tensor &s);

} // namespace infinicore::op
