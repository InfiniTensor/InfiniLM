#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Sinh {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor sinh(Tensor input);
void sinh_(Tensor output, Tensor input);

} // namespace infinicore::op
