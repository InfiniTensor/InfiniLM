#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Reciprocal {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor y, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor reciprocal(Tensor x);
void reciprocal_(Tensor y, Tensor x);
} // namespace infinicore::op
