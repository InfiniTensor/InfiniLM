#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class GeluTanh {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor gelu_tanh(Tensor input);
void gelu_tanh_(Tensor output, Tensor input);
} // namespace infinicore::op
