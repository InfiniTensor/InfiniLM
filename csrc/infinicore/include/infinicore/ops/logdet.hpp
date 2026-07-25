#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Logdet, Tensor, const Tensor &);

Tensor logdet(const Tensor &x);
void logdet_(Tensor y, const Tensor &x);

} // namespace infinicore::op
