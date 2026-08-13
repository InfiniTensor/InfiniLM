#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class FloorDivide {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor c, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor floor_divide(Tensor a, Tensor b);
void floor_divide_(Tensor c, Tensor a, Tensor b);
} // namespace infinicore::op
