#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <string>

namespace infinicore::op {

class BinaryCrossEntropyWithLogits {
public:
    /**
     * @brief BCEWithLogits 算子的函数原型
     * 参数顺序: out, logits, target, weight, pos_weight, reduction
     */
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, std::string);

    static void execute(Tensor out,
                        Tensor logits,
                        Tensor target,
                        Tensor weight,
                        Tensor pos_weight,
                        std::string reduction);

    static common::OpDispatcher<schema> &dispatcher();
};

/**
 * @brief 非原地操作接口 (Out-of-place)
 */
Tensor binary_cross_entropy_with_logits(Tensor logits,
                                        Tensor target,
                                        Tensor weight = {},
                                        Tensor pos_weight = {},
                                        std::string reduction = "mean");

/**
 * @brief 显式指定输出张量的接口
 */
void binary_cross_entropy_with_logits_(Tensor out,
                                       Tensor logits,
                                       Tensor target,
                                       Tensor weight,
                                       Tensor pos_weight,
                                       std::string reduction);

} // namespace infinicore::op
