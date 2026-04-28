#pragma once

#include "infinicore.h"

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Interpolate, Tensor, const Tensor &, std::string, std::vector<int64_t>, std::vector<double>, int);

__export Tensor interpolate(const Tensor &input,
                            std::string mode,
                            std::vector<int64_t> size,
                            std::vector<double> scale_factor,
                            int align_corners);

__export void interpolate_(Tensor out,
                           const Tensor &input,
                           std::string mode,
                           std::vector<int64_t> size,
                           std::vector<double> scale_factor,
                           int align_corners);

} // namespace infinicore::op
