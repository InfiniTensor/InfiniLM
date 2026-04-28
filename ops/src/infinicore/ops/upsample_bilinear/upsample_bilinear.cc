#include "infinicore/ops/upsample_bilinear.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<UpsampleBilinear::schema> &UpsampleBilinear::dispatcher() {
    static common::OpDispatcher<UpsampleBilinear::schema> dispatcher_;
    return dispatcher_;
};

void UpsampleBilinear::execute(Tensor output, Tensor input, bool align_corners) {
    dispatcher().lookup(context::getDevice().getType())(output, input, align_corners);
}

// 3. 函数式接口
Tensor upsample_bilinear(Tensor input, std::vector<int64_t> output_size, bool align_corners) {
    // 构造输出 Shape
    // 假设 input 是 (N, C, H_in, W_in) 或 (C, H_in, W_in)
    // output_size 通常只包含 (H_out, W_out)
    Shape input_shape = input->shape();
    size_t ndim = input_shape.size();

    Shape output_shape = input_shape;

    // 更新最后两个维度为 output_size 指定的大小
    if (output_size.size() == 2 && ndim >= 2) {
        output_shape[ndim - 2] = output_size[0];
        output_shape[ndim - 1] = output_size[1];
    }

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());

    upsample_bilinear_(output, input, align_corners);
    return output;
}

void upsample_bilinear_(Tensor output, Tensor input, bool align_corners) {
    UpsampleBilinear::execute(output, input, align_corners);
}

} // namespace infinicore::op
