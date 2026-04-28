#pragma once

#include "infinicore.h"

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(GaussianNllLoss, Tensor, const Tensor &, const Tensor &, const Tensor &, bool, double, int);

__export Tensor gaussian_nll_loss(const Tensor &input,
                                  const Tensor &target,
                                  const Tensor &var,
                                  bool full = false,
                                  double eps = 1e-6,
                                  int reduction = 1);

__export void gaussian_nll_loss_(Tensor out,
                                 const Tensor &input,
                                 const Tensor &target,
                                 const Tensor &var,
                                 bool full = false,
                                 double eps = 1e-6,
                                 int reduction = 1);

} // namespace infinicore::op
