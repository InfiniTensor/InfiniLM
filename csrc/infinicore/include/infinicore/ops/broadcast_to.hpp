#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(BroadcastTo, Tensor, Tensor);

Tensor broadcast_to(Tensor x, const std::vector<int64_t> &shape);
void broadcast_to_(Tensor y, Tensor x);

} // namespace infinicore::op
