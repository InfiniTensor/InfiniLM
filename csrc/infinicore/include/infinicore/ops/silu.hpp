#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Silu {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor silu(Tensor input);
void silu_(Tensor output, Tensor input);
} // namespace infinicore::op
