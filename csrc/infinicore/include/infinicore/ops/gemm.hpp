#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Gemm, Tensor, const Tensor &, const Tensor &, float, float);

Tensor gemm(const Tensor &a, const Tensor &b, float alpha = 1.0f, float beta = 0.0f);
void gemm_(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta);

} // namespace infinicore::op
