#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <vector>

namespace infinicore::op {

class UpsampleNearest {
public:
    // Schema signature: output(out), input
    // Note: Scales are inferred from output.shape / input.shape
    using schema = void (*)(Tensor, Tensor);

    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

// Functional API: Returns the result tensor
// Requires output_size to calculate the shape of the result tensor
Tensor upsample_nearest(Tensor input, const std::vector<int64_t> &output_size);

// In-place/Output-provided API
void upsample_nearest_(Tensor output, Tensor input);

} // namespace infinicore::op
