#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Fmin {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor c, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor fmin(Tensor a, Tensor b);
void fmin_(Tensor c, Tensor a, Tensor b);

} // namespace infinicore::op
