#include "infinicore/ops/triplet_margin_loss.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<TripletMarginLoss::schema> &TripletMarginLoss::dispatcher() {
    static common::OpDispatcher<TripletMarginLoss::schema> dispatcher_;
    return dispatcher_;
};

void TripletMarginLoss::execute(Tensor output, Tensor anchor, Tensor positive, Tensor negative, float margin, int64_t p, float eps, bool swap, int64_t reduction) {
    dispatcher().lookup(context::getDevice().getType())(output, anchor, positive, negative, margin, p, eps, swap, reduction);
}

// 3. 函数式接口
Tensor triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin, int64_t p, float eps, bool swap, int64_t reduction) {
    Shape output_shape;
    if (reduction == 0) { // None
        // TripletMarginLoss 输入通常为 (N, D)，reduction='none' 时输出为 (N)
        // 取第 0 维作为 Batch Size
        output_shape = {anchor->shape()[0]};
    } else {
        output_shape = {}; // Scalar
    }

    // 使用 anchor 的属性创建输出 Tensor
    auto output = Tensor::empty(output_shape, anchor->dtype(), anchor->device());

    triplet_margin_loss_(output, anchor, positive, negative, margin, p, eps, swap, reduction);
    return output;
}

void triplet_margin_loss_(Tensor output, Tensor anchor, Tensor positive, Tensor negative, float margin, int64_t p, float eps, bool swap, int64_t reduction) {
    TripletMarginLoss::execute(output, anchor, positive, negative, margin, p, eps, swap, reduction);
}

} // namespace infinicore::op
