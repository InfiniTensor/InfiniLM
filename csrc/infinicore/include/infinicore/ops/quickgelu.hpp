#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class QuickGelu {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor quick_gelu(Tensor input);
void quick_gelu_(Tensor output, Tensor input);
} // namespace infinicore::op
