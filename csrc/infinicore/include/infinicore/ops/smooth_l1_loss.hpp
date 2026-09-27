#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class SmoothL1Loss {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, float, int64_t);

    static void execute(Tensor output, Tensor input, Tensor target, float beta, int64_t reduction);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor smooth_l1_loss(Tensor input, Tensor target, float beta = 1.0f, int64_t reduction = 1);
void smooth_l1_loss_(Tensor output, Tensor input, Tensor target, float beta, int64_t reduction);

} // namespace infinicore::op
