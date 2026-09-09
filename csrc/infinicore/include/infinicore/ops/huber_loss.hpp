#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class HuberLoss {
public:
    // Schema: output, input, target, delta, reduction
    using schema = void (*)(Tensor, Tensor, Tensor, float, int64_t);

    static void execute(Tensor output, Tensor input, Tensor target, float delta, int64_t reduction);
    static common::OpDispatcher<schema> &dispatcher();
};

// delta 默认为 1.0f，reduction 默认为 1 (MEAN)
Tensor huber_loss(Tensor input, Tensor target, float delta = 1.0f, int64_t reduction = 1);
void huber_loss_(Tensor output, Tensor input, Tensor target, float delta, int64_t reduction);

} // namespace infinicore::op
