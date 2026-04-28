#pragma once

#include "infinicore.h"

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Relu6, Tensor, const Tensor &);

__export Tensor relu6(const Tensor &input);
__export void relu6_(Tensor out, const Tensor &input);

} // namespace infinicore::op
