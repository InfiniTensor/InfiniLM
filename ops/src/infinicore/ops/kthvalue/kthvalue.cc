#include "infinicore/ops/kthvalue.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<Kthvalue::schema> &Kthvalue::dispatcher() {
    static common::OpDispatcher<Kthvalue::schema> dispatcher_;
    return dispatcher_;
};

void Kthvalue::execute(Tensor values, Tensor indices, Tensor input, int64_t k, int64_t dim, bool keepdim) {
    dispatcher().lookup(context::getDevice().getType())(values, indices, input, k, dim, keepdim);
}

// 3. 函数式接口
std::tuple<Tensor, Tensor> kthvalue(Tensor input, int64_t k, int64_t dim, bool keepdim) {
    auto input_shape = input->shape();
    int64_t ndim = input_shape.size();

    // 处理负数维度
    if (dim < 0) {
        dim += ndim;
    }

    Shape output_shape;
    if (keepdim) {
        output_shape = input_shape;
        output_shape[dim] = 1;
    } else {
        output_shape.reserve(ndim - 1);
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != dim) {
                output_shape.push_back(input_shape[i]);
            }
        }
    }

    // values 与 input 类型一致
    auto values = Tensor::empty(output_shape, input->dtype(), input->device());
    auto indices = Tensor::empty(output_shape, DataType::I64, input->device());
    kthvalue_(values, indices, input, k, dim, keepdim);
    return std::make_tuple(values, indices);
}

void kthvalue_(Tensor values, Tensor indices, Tensor input, int64_t k, int64_t dim, bool keepdim) {
    Kthvalue::execute(values, indices, input, k, dim, keepdim);
}

} // namespace infinicore::op
