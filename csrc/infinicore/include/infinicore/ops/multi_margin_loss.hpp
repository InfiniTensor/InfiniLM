#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class MultiMarginLoss {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, int64_t, float, int64_t);

    static void execute(Tensor output, Tensor input, Tensor target, Tensor weight, int64_t p, float margin, int64_t reduction);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor multi_margin_loss(Tensor input, Tensor target, Tensor weight = {}, int64_t p = 1, float margin = 1.0f, int64_t reduction = 1);
void multi_margin_loss_(Tensor output, Tensor input, Tensor target, Tensor weight, int64_t p, float margin, int64_t reduction);

} // namespace infinicore::op
