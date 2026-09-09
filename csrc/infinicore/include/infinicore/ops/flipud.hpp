#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Flipud {
public:
    // Schema signature: (Output, Input)
    using schema = void (*)(Tensor, Tensor);

    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};
Tensor flipud(Tensor input);
void flipud_(Tensor output, Tensor input);

} // namespace infinicore::op
