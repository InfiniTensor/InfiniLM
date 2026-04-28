#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Gelu {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor gelu(Tensor input);
void gelu_(Tensor output, Tensor input);
} // namespace infinicore::op
