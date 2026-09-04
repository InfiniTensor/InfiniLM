#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class AvgPool1d {
public:
    using schema = void (*)(Tensor, Tensor, size_t, size_t, size_t);
    static void execute(Tensor output, Tensor input, size_t kernel_size, size_t stride, size_t padding);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor avg_pool1d(Tensor input, size_t kernel_size, size_t stride = 0, size_t padding = 0);
void avg_pool1d_(Tensor output, Tensor input, size_t kernel_size, size_t stride = 0, size_t padding = 0);

} // namespace infinicore::op
