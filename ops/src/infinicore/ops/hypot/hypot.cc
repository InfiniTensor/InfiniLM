#include "infinicore/ops/hypot.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<Hypot::schema> &Hypot::dispatcher() {
    static common::OpDispatcher<Hypot::schema> dispatcher_;
    return dispatcher_;
};

void Hypot::execute(Tensor output, Tensor input_a, Tensor input_b) {
    // lookup 需要传入设备类型，然后调用返回的函数指针
    dispatcher().lookup(context::getDevice().getType())(output, input_a, input_b);
}
Tensor hypot(Tensor input_a, Tensor input_b) {
    auto output = Tensor::empty(input_a->shape(), input_a->dtype(), input_a->device());

    hypot_(output, input_a, input_b);
    return output;
}
void hypot_(Tensor output, Tensor input_a, Tensor input_b) {
    Hypot::execute(output, input_a, input_b);
}

} // namespace infinicore::op
