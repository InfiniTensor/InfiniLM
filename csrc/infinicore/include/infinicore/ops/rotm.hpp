#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Rotm {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor x, Tensor y, Tensor param);
    static common::OpDispatcher<schema> &dispatcher();
};

void rotm_(Tensor x, Tensor y, Tensor param);

} // namespace infinicore::op
