#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(BitwiseRightShift, Tensor, const Tensor &, const Tensor &);

Tensor bitwise_right_shift(const Tensor &input, const Tensor &other);
void bitwise_right_shift_(Tensor out, const Tensor &input, const Tensor &other);

} // namespace infinicore::op
