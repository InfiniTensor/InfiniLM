#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Asinh {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor y, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor asinh(Tensor x);
void asinh_(Tensor y, Tensor x);
} // namespace infinicore::op
