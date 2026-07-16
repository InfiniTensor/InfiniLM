#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Swap, Tensor, Tensor);

void swap_(Tensor x, Tensor y);

} // namespace infinicore::op
