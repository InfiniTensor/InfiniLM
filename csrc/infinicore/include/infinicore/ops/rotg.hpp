#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Rotg {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor);
    static void execute(Tensor x, Tensor y, Tensor c, Tensor s);
    static common::OpDispatcher<schema> &dispatcher();
};

void rotg_(Tensor x, Tensor y, Tensor c, Tensor s);

} // namespace infinicore::op
