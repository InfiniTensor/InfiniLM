#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Selu {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor selu(Tensor input);
void selu_(Tensor output, Tensor input);

} // namespace infinicore::op
