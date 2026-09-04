#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(Ones, Tensor);

void ones_(Tensor output);
} // namespace infinicore::op
