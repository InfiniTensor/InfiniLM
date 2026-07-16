#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <vector>

namespace infinicore::op {

class Unfold {
public:
    // schema: output, input, kernel_sizes, dilations, paddings, strides
    using schema = void (*)(Tensor, Tensor, const std::vector<int64_t> &, const std::vector<int64_t> &, const std::vector<int64_t> &, const std::vector<int64_t> &);

    static void execute(Tensor output, Tensor input,
                        const std::vector<int64_t> &kernel_sizes,
                        const std::vector<int64_t> &dilations,
                        const std::vector<int64_t> &paddings,
                        const std::vector<int64_t> &strides);
    static common::OpDispatcher<schema> &dispatcher();
};

// Functional API
Tensor unfold(Tensor input,
              std::vector<int64_t> kernel_sizes,
              std::vector<int64_t> dilations,
              std::vector<int64_t> paddings,
              std::vector<int64_t> strides);

void unfold_(Tensor output, Tensor input,
             std::vector<int64_t> kernel_sizes,
             std::vector<int64_t> dilations,
             std::vector<int64_t> paddings,
             std::vector<int64_t> strides);

} // namespace infinicore::op
