#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Prelu, Tensor, const Tensor &, const Tensor &);

Tensor prelu(const Tensor &input, const Tensor &weight);
void prelu_(Tensor out, const Tensor &input, const Tensor &weight);

} // namespace infinicore::op
