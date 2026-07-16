#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Softmax {
public:
    using schema = void (*)(Tensor, Tensor, int);
    static void execute(Tensor output, Tensor input, int axis);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor softmax(Tensor input, int axis = -1);
void softmax_(Tensor output, Tensor input, int axis = -1);
} // namespace infinicore::op
