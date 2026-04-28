#include "infinicore/ops/triplet_margin_with_distance_loss.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<TripletMarginWithDistanceLoss::schema> &TripletMarginWithDistanceLoss::dispatcher() {
    static common::OpDispatcher<TripletMarginWithDistanceLoss::schema> dispatcher_;
    return dispatcher_;
};

void TripletMarginWithDistanceLoss::execute(Tensor output, Tensor anchor, Tensor positive, Tensor negative, double margin, bool swap, int64_t reduction) {
    dispatcher().lookup(context::getDevice().getType())(output, anchor, positive, negative, margin, swap, reduction);
}

// 3. 函数式接口
Tensor triplet_margin_with_distance_loss(Tensor anchor, Tensor positive, Tensor negative, double margin, bool swap, int64_t reduction) {
    Shape out_shape;

    // 推断输出形状
    if (reduction == 0) {
        // Reduction::None -> 输出形状取决于输入的广播结果
        out_shape = anchor->shape();
    } else {
        // Reduction::Mean 或 Reduction::Sum -> 输出为标量
        out_shape = {};
    }

    auto output = Tensor::empty(out_shape, anchor->dtype(), anchor->device());

    triplet_margin_with_distance_loss_(output, anchor, positive, negative, margin, swap, reduction);
    return output;
}

void triplet_margin_with_distance_loss_(Tensor output, Tensor anchor, Tensor positive, Tensor negative, double margin, bool swap, int64_t reduction) {
    TripletMarginWithDistanceLoss::execute(output, anchor, positive, negative, margin, swap, reduction);
}

} // namespace infinicore::op
