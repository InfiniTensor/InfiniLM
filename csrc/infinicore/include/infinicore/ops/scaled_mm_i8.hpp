#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(I8Gemm, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, std::optional<Tensor>);

void scaled_mm_i8_(Tensor c, const Tensor &a_p, const Tensor &a_s, const Tensor &b_p, const Tensor &b_s, std::optional<Tensor> bias);
} // namespace infinicore::op
