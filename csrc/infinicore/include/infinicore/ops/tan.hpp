#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Tan {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor tan(Tensor input);
void tan_(Tensor output, Tensor input);

} // namespace infinicore::op
