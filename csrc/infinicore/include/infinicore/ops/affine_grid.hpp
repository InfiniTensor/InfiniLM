#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <vector>

namespace infinicore::op {

class AffineGrid {
public:
    using schema = void (*)(Tensor, Tensor, bool);
    static void execute(Tensor output, Tensor theta, bool align_corners);
    static common::OpDispatcher<schema> &dispatcher();
};
Tensor affine_grid(Tensor theta, const std::vector<int64_t> &size, bool align_corners = false);

} // namespace infinicore::op
