#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Hardswish {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor hardswish(Tensor input);
void hardswish_(Tensor output, Tensor input);

} // namespace infinicore::op
