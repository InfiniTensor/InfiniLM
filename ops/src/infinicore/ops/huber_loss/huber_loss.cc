#include "infinicore/ops/huber_loss.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<HuberLoss::schema> &HuberLoss::dispatcher() {
    static common::OpDispatcher<HuberLoss::schema> dispatcher_;
    return dispatcher_;
};

void HuberLoss::execute(Tensor output, Tensor input, Tensor target, float delta, int64_t reduction) {
    dispatcher().lookup(context::getDevice().getType())(output, input, target, delta, reduction);
}

// 3. 函数式接口
Tensor huber_loss(Tensor input, Tensor target, float delta, int64_t reduction) {
    Shape output_shape;
    if (reduction == 0) { // None
        // HuberLoss 是 Element-wise 的，reduction='none' 时输出形状通常与输入一致
        output_shape = input->shape();
    } else {
        output_shape = {}; // Scalar
    }

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());

    huber_loss_(output, input, target, delta, reduction);
    return output;
}

void huber_loss_(Tensor output, Tensor input, Tensor target, float delta, int64_t reduction) {
    HuberLoss::execute(output, input, target, delta, reduction);
}

} // namespace infinicore::op
