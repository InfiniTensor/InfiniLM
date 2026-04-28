#include "infinicore/ops/take.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<Take::schema> &Take::dispatcher() {
    static common::OpDispatcher<Take::schema> dispatcher_;
    return dispatcher_;
};

// 2. Execute 实现：查找对应设备的核函数并执行
void Take::execute(Tensor output, Tensor input, Tensor indices) {
    dispatcher().lookup(context::getDevice().getType())(output, input, indices);
}

Tensor take(Tensor input, Tensor indices) {
    // 【关键区别】Take 的输出形状取决于 indices 的形状，但数据类型取决于 input
    auto output = Tensor::empty(indices->shape(), input->dtype(), input->device());

    take_(output, input, indices);
    return output;
}

void take_(Tensor output, Tensor input, Tensor indices) {
    Take::execute(output, input, indices);
}

} // namespace infinicore::op
