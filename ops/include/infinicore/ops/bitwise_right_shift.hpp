#pragma once

#include "infinicore.h"

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(BitwiseRightShift, Tensor, const Tensor &, const Tensor &);

__export Tensor bitwise_right_shift(const Tensor &input, const Tensor &other);
__export void bitwise_right_shift_(Tensor out, const Tensor &input, const Tensor &other);

} // namespace infinicore::op
