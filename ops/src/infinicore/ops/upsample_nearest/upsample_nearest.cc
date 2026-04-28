#include "infinicore/ops/upsample_nearest.hpp"
#include <stdexcept>
namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<UpsampleNearest::schema> &UpsampleNearest::dispatcher() {
    static common::OpDispatcher<UpsampleNearest::schema> dispatcher_;
    return dispatcher_;
};

void UpsampleNearest::execute(Tensor output, Tensor input) {
    dispatcher().lookup(context::getDevice().getType())(output, input);
}

// 3. 函数式接口
Tensor upsample_nearest(Tensor input, const std::vector<int64_t> &output_size) {
    Shape input_shape = input->shape();
    size_t ndim = input_shape.size();

    // 校验
    if (ndim < 3 || ndim > 4) {
        if (ndim != 3 && ndim != 4) {
            throw std::runtime_error("upsample_nearest: Only supports 3D (N,C,W) or 4D (N,C,H,W) input");
        }
    }

    Shape output_shape = input_shape;

    if (ndim == 3) {
        // [N, C, W]
        // output_size 可能是 [W_out] (size=1) 或者 [1, W_out] (size=2)
        int64_t target_w = 0;
        if (output_size.size() == 1) {
            target_w = output_size[0];
        } else if (output_size.size() == 2) {
            target_w = output_size[1];
        } else {
            throw std::runtime_error("upsample_nearest: output_size for 3D input must be [w] or [1, w]");
        }
        output_shape[2] = target_w;

    } else if (ndim == 4) {
        // [N, C, H, W]
        if (output_size.size() != 2) {
            throw std::runtime_error("upsample_nearest: output_size for 4D input must be [h, w]");
        }
        output_shape[2] = output_size[0];
        output_shape[3] = output_size[1];
    }

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());

    upsample_nearest_(output, input);
    return output;
}

void upsample_nearest_(Tensor output, Tensor input) {
    UpsampleNearest::execute(output, input);
}

} // namespace infinicore::op
