#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Relu {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor relu(Tensor input);
void relu_(Tensor output, Tensor input);
} // namespace infinicore::op
