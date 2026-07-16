#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(PerTensorDequantI8, Tensor, const Tensor &, const Tensor &, const Tensor &);

void per_tensor_dequant_i8_(Tensor x, const Tensor &x_packed, const Tensor &x_scale, const Tensor &x_zero);
} // namespace infinicore::op
