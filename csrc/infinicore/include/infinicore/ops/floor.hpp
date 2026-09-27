#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Floor {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor floor(Tensor input);
void floor_(Tensor output, Tensor input);
} // namespace infinicore::op
