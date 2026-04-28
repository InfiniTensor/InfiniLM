#include "infinicore/ops/multi_margin_loss.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<MultiMarginLoss::schema> &MultiMarginLoss::dispatcher() {
    static common::OpDispatcher<MultiMarginLoss::schema> dispatcher_;
    return dispatcher_;
};

void MultiMarginLoss::execute(Tensor output, Tensor input, Tensor target, Tensor weight, int64_t p, float margin, int64_t reduction) {
    dispatcher().lookup(context::getDevice().getType())(output, input, target, weight, p, margin, reduction);
}

// 3. 函数式接口
Tensor multi_margin_loss(Tensor input, Tensor target, Tensor weight, int64_t p, float margin, int64_t reduction) {
    Shape output_shape;
    if (reduction == 0) { // None
        output_shape = {input->shape()[0]};
    } else {
        output_shape = {}; // Scalar
    }

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());

    multi_margin_loss_(output, input, target, weight, p, margin, reduction);
    return output;
}

void multi_margin_loss_(Tensor output, Tensor input, Tensor target, Tensor weight, int64_t p, float margin, int64_t reduction) {
    MultiMarginLoss::execute(output, input, target, weight, p, margin, reduction);
}

} // namespace infinicore::op
