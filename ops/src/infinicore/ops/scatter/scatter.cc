#include "infinicore/ops/scatter.hpp"

namespace infinicore::op {

common::OpDispatcher<Scatter::schema> &Scatter::dispatcher() {
    static common::OpDispatcher<Scatter::schema> dispatcher_;
    return dispatcher_;
};

void Scatter::execute(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor src, int64_t reduction) {
    dispatcher().lookup(context::getDevice().getType())(output, input, dim, index, src, reduction);
}

Tensor scatter(Tensor input, int64_t dim, Tensor index, Tensor src, int64_t reduction) {
    // 创建与 input 形状、数据类型、设备一致的 Output Tensor
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    scatter_(output, input, dim, index, src, reduction);

    return output;
}

void scatter_(Tensor output, Tensor input, int64_t dim, Tensor index, Tensor src, int64_t reduction) {
    Scatter::execute(output, input, dim, index, src, reduction);
}

} // namespace infinicore::op
