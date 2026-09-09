#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(PerTensorQuantI8, const Tensor &, Tensor, Tensor, Tensor, bool);

void per_tensor_quant_i8_(const Tensor &x, Tensor x_packed, Tensor x_scale, Tensor x_zero, bool is_static);

Tensor per_tensor_quant_i8(const Tensor &x, Tensor x_scale, Tensor x_zero, bool is_static);
} // namespace infinicore::op
