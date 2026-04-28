#include "infinicore/ops/unfold.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<Unfold::schema> &Unfold::dispatcher() {
    static common::OpDispatcher<Unfold::schema> dispatcher_;
    return dispatcher_;
};

// 2. Execute 方法实现
void Unfold::execute(Tensor output, Tensor input,
                     const std::vector<int64_t> &kernel_sizes,
                     const std::vector<int64_t> &dilations,
                     const std::vector<int64_t> &paddings,
                     const std::vector<int64_t> &strides) {
    dispatcher().lookup(context::getDevice().getType())(output, input, kernel_sizes, dilations, paddings, strides);
}

// 3. 函数式接口
Tensor unfold(Tensor input,
              std::vector<int64_t> kernel_sizes,
              std::vector<int64_t> dilations,
              std::vector<int64_t> paddings,
              std::vector<int64_t> strides) {

    // 基础维度校验与获取
    const auto &input_shape = input->shape();
    int64_t n_dim = input->ndim();
    int64_t spatial_dims = n_dim - 2; // N, C, D1, D2... -> spatial starts at 2
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];

    // 计算 dim 1: C * kernel_sizes[0] * kernel_sizes[1] ...
    int64_t output_dim1 = C;
    for (auto k : kernel_sizes) {
        output_dim1 *= k;
    }

    int64_t L = 1;
    for (int i = 0; i < spatial_dims; ++i) {
        int64_t input_dim = input_shape[i + 2];
        int64_t k = kernel_sizes[i];
        int64_t p = paddings[i];
        int64_t d = dilations[i];
        int64_t s = strides[i];

        // 公式: out = floor((in + 2*p - d*(k-1) - 1) / s + 1)
        int64_t output_spatial = (input_dim + 2 * p - d * (k - 1) - 1) / s + 1;
        L *= output_spatial;
    }
    Shape output_shape = {
        static_cast<size_t>(N),
        static_cast<size_t>(output_dim1),
        static_cast<size_t>(L)};

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());

    unfold_(output, input, kernel_sizes, dilations, paddings, strides);
    return output;
}

// 4. In-place / 显式输出接口
void unfold_(Tensor output, Tensor input,
             std::vector<int64_t> kernel_sizes,
             std::vector<int64_t> dilations,
             std::vector<int64_t> paddings,
             std::vector<int64_t> strides) {
    Unfold::execute(output, input, kernel_sizes, dilations, paddings, strides);
}

} // namespace infinicore::op
