#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Zeros, Tensor);

void zeros_(Tensor output);

} // namespace infinicore::op
