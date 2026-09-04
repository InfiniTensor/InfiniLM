#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Hypot {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);

    static void execute(Tensor output, Tensor input_a, Tensor input_b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor hypot(Tensor input_a, Tensor input_b);

void hypot_(Tensor output, Tensor input_a, Tensor input_b);
} // namespace infinicore::op
