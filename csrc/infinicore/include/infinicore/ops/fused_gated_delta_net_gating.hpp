#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <utility>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(FusedGatedDeltaNetGating,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          float,
                          float);

std::pair<Tensor, Tensor> fused_gated_delta_net_gating(const Tensor &A_log,
                                                       const Tensor &a,
                                                       const Tensor &b,
                                                       const Tensor &dt_bias,
                                                       float beta = 1.0f,
                                                       float threshold = 20.0f);

void fused_gated_delta_net_gating_(Tensor g,
                                   Tensor beta_output,
                                   const Tensor &A_log,
                                   const Tensor &a,
                                   const Tensor &b,
                                   const Tensor &dt_bias,
                                   float beta = 1.0f,
                                   float threshold = 20.0f);

} // namespace infinicore::op
