#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <vector>

namespace infinicore::op {

class UpsampleBilinear {
public:
    // Schema signature: output, input, align_corners
    using schema = void (*)(Tensor, Tensor, bool);

    static void execute(Tensor output, Tensor input, bool align_corners);
    static common::OpDispatcher<schema> &dispatcher();
};

// 需要传入 output_size (如 {H_out, W_out} 或 {N, C, H_out, W_out}) 来决定新 Tensor 的形状
Tensor upsample_bilinear(Tensor input, std::vector<int64_t> output_size, bool align_corners = false);
void upsample_bilinear_(Tensor output, Tensor input, bool align_corners);

} // namespace infinicore::op
