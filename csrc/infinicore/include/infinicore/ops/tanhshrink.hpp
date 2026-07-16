#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Tanhshrink {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor tanhshrink(Tensor input);
void tanhshrink_(Tensor output, Tensor input);
} // namespace infinicore::op
