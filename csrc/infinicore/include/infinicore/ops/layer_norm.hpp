#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(LayerNorm, Tensor, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &, float);

Tensor layer_norm(const Tensor &x, const Tensor &weight, const Tensor &bias, float epsilon = 1e-5f);
void layer_norm_(Tensor y, Tensor standardization, Tensor std_deviation, const Tensor &x, const Tensor &weight, const Tensor &bias, float epsilon = 1e-5f);
void layer_norm_(Tensor y, const Tensor &x, const Tensor &weight, const Tensor &bias, float epsilon = 1e-5f);
void layer_norm_for_pybind(Tensor y, const Tensor &x, const Tensor &weight, const Tensor &bias, float epsilon = 1e-5f);

} // namespace infinicore::op
