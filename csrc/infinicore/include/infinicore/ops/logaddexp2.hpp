#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class LogAddExp2 {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor c, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor logaddexp2(Tensor a, Tensor b);
void logaddexp2_(Tensor c, Tensor a, Tensor b);

} // namespace infinicore::op
