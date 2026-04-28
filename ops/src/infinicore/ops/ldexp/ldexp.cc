#include "infinicore/ops/ldexp.hpp"
#include <algorithm> // for std::max

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<Ldexp::schema> &Ldexp::dispatcher() {
    static common::OpDispatcher<Ldexp::schema> dispatcher_;
    return dispatcher_;
};

void Ldexp::execute(Tensor output, Tensor input, Tensor other) {
    dispatcher().lookup(context::getDevice().getType())(output, input, other);
}

// 2. 函数式接口
Tensor ldexp(Tensor input, Tensor other) {
    // 计算广播后的输出形状 (Broadcasting Logic)
    const auto &shape_a = input->shape();
    const auto &shape_b = other->shape();

    size_t ndim_a = shape_a.size();
    size_t ndim_b = shape_b.size();
    size_t ndim_out = std::max(ndim_a, ndim_b);

    Shape output_shape(ndim_out);

    // 从后往前对齐维度进行广播检查
    for (size_t i = 0; i < ndim_out; ++i) {
        // 获取对应的维度大小，若越界则视为 1 (右对齐)
        int64_t dim_a = (i >= ndim_out - ndim_a) ? shape_a[i - (ndim_out - ndim_a)] : 1;
        int64_t dim_b = (i >= ndim_out - ndim_b) ? shape_b[i - (ndim_out - ndim_b)] : 1;
        output_shape[i] = std::max(dim_a, dim_b);
    }

    // 分配输出 Tensor
    // ldexp 的输出类型通常跟随 input (尾数)，设备跟随 input
    auto output = Tensor::empty(output_shape, input->dtype(), input->device());

    // 调用 Explicit output 接口
    ldexp_(output, input, other);

    return output;
}

// 3. Explicit Output 接口
void ldexp_(Tensor output, Tensor input, Tensor other) {
    Ldexp::execute(output, input, other);
}

} // namespace infinicore::op
