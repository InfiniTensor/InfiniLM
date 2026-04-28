#include "infinicore/ops/vander.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<Vander::schema> &Vander::dispatcher() {
    static common::OpDispatcher<Vander::schema> dispatcher_;
    return dispatcher_;
};

// 2. Execute 方法实现
void Vander::execute(Tensor output, Tensor input, int64_t N, bool increasing) {
    dispatcher().lookup(context::getDevice().getType())(output, input, N, increasing);
}

// 3. 函数式接口
Tensor vander(Tensor input, int64_t N, bool increasing) {
    int64_t input_size = input->shape()[0];
    int64_t cols = (N > 0) ? N : input_size;
    Shape output_shape = {
        static_cast<size_t>(input_size),
        static_cast<size_t>(cols)};

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());

    vander_(output, input, N, increasing);
    return output;
}

// 4. In-place / 显式输出接口
void vander_(Tensor output, Tensor input, int64_t N, bool increasing) {
    Vander::execute(output, input, N, increasing);
}

} // namespace infinicore::op
