#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Topksoftmax, Tensor, Tensor, const Tensor &, const size_t, const int);

void topksoftmax(Tensor values, Tensor indices, const Tensor &x, const size_t topk, const int norm = 0);

} // namespace infinicore::op
