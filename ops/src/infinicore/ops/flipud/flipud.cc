#include "infinicore/ops/flipud.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<Flipud::schema> &Flipud::dispatcher() {
    static common::OpDispatcher<Flipud::schema> dispatcher_;
    return dispatcher_;
}

// 2. 静态执行函数
void Flipud::execute(Tensor output, Tensor input) {
    dispatcher().lookup(context::getDevice().getType())(output, input);
}
Tensor flipud(Tensor input) {
    // Flipud 操作不改变张量的形状和数据类型
    // Output shape == Input shape
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());

    flipud_(output, input);
    return output;
}
void flipud_(Tensor output, Tensor input) {
    Flipud::execute(output, input);
}

} // namespace infinicore::op
