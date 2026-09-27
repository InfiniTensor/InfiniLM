#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <string>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Pad, Tensor, const Tensor &, const std::vector<int> &, const std::string &, double);

Tensor pad(const Tensor &x,
           const std::vector<int> &pad,
           const std::string &mode = "constant",
           double value = 0.0);

void pad_(Tensor y,
          const Tensor &x,
          const std::vector<int> &pad,
          const std::string &mode = "constant",
          double value = 0.0);

} // namespace infinicore::op
