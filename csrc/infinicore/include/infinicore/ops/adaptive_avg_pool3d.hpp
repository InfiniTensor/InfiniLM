#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class AdaptiveAvgPool3D {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor y, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor adaptive_avg_pool3d(Tensor x, std::vector<size_t> output_size);
void adaptive_avg_pool3d_(Tensor y, Tensor x);
} // namespace infinicore::op
