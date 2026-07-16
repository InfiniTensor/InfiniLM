#pragma once
#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(PerChannelQuantI8, const Tensor &, Tensor, Tensor);

void per_channel_quant_i8_(const Tensor &x, Tensor x_packed, Tensor x_scale);
} // namespace infinicore::op
