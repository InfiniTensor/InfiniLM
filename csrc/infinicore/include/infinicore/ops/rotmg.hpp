#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Rotmg {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor);
    static void execute(Tensor d1, Tensor d2, Tensor x1, Tensor y1, Tensor param);
    static common::OpDispatcher<schema> &dispatcher();
};

void rotmg_(Tensor d1, Tensor d2, Tensor x1, Tensor y1, Tensor param);

} // namespace infinicore::op
