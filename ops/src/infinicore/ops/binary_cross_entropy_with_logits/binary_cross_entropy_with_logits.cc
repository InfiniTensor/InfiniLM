#include "infinicore/ops/binary_cross_entropy_with_logits.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

// 静态调度器实例化
common::OpDispatcher<BinaryCrossEntropyWithLogits::schema> &BinaryCrossEntropyWithLogits::dispatcher() {
    static common::OpDispatcher<BinaryCrossEntropyWithLogits::schema> dispatcher_;
    return dispatcher_;
};

/**
 * 执行核心逻辑：设备校验、上下文设置与后端分发
 */
void BinaryCrossEntropyWithLogits::execute(Tensor out, Tensor logits, Tensor target, Tensor weight, Tensor pos_weight, std::string reduction) {
    // 1. 校验所有已定义的 Tensor 是否在同一设备上
    // 使用宏或循环校验 logits, target, out 以及可选的 weight/pos_weight
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, logits, target);
    if (weight) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, weight);
    }
    if (pos_weight) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, pos_weight);
    }

    // 2. 设置当前设备上下文
    infinicore::context::setDevice(out->device());

    // 3. 根据设备类型查找并执行具体的后端实现（如 CUDA 或 CPU 实现）
    dispatcher().lookup(out->device().getType())(out, logits, target, weight, pos_weight, reduction);
}

/**
 * Out-of-place 接口：根据 reduction 自动创建输出 Tensor
 */
Tensor binary_cross_entropy_with_logits(Tensor logits, Tensor target, Tensor weight, Tensor pos_weight, std::string reduction) {
    std::vector<uint64_t> out_shape;

    // 1. 根据归约方式确定输出形状
    if (reduction == "none") {
        // 不归约，形状与输入 logits 一致
        auto in_shape = logits->shape();
        for (auto dim : in_shape) {
            out_shape.push_back(static_cast<uint64_t>(dim));
        }
    } else {
        // mean 或 sum 归约，输出为标量 (空 shape 向量表示 0-dim tensor)
        out_shape = {};
    }

    // 2. 创建输出 Tensor
    auto out = Tensor::empty(out_shape, logits->dtype(), logits->device());

    // 3. 调用显式接口执行计算
    binary_cross_entropy_with_logits_(out, logits, target, weight, pos_weight, reduction);

    return out;
}

/**
 * 显式指定输出接口
 */
void binary_cross_entropy_with_logits_(Tensor out, Tensor logits, Tensor target, Tensor weight, Tensor pos_weight, std::string reduction) {
    BinaryCrossEntropyWithLogits::execute(out, logits, target, weight, pos_weight, reduction);
}

} // namespace infinicore::op
