#include "infinicore/ops/log_softmax.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<LogSoftmax::schema> &LogSoftmax::dispatcher() {
    static common::OpDispatcher<LogSoftmax::schema> dispatcher_;
    return dispatcher_;
};

void LogSoftmax::execute(Tensor output, Tensor input, int64_t dim) {
    dispatcher().lookup(context::getDevice().getType())(output, input, dim);
}

// 3. 函数式接口
Tensor log_softmax(Tensor input, int64_t dim) {
    int64_t ndim = input->shape().size();

    // 处理负数维度
    if (dim < 0) {
        dim += ndim;
    }

    // LogSoftmax 输出形状与输入一致，dtype 与 input 一致
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    log_softmax_(output, input, dim);
    return output;
}

void log_softmax_(Tensor output, Tensor input, int64_t dim) {
    LogSoftmax::execute(output, input, dim);
}

} // namespace infinicore::op
