#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Take {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);

    static void execute(Tensor output, Tensor input, Tensor indices);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor take(Tensor input, Tensor indices);

void take_(Tensor output, Tensor input, Tensor indices);

} // namespace infinicore::op
