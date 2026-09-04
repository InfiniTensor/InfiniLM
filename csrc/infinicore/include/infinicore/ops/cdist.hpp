#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Cdist {
public:
    /**
     * @brief 成对距离计算算子 (Pairwise distance)
     * schema: out (M, N), x1 (M, D), x2 (N, D), p (norm degree)
     */
    using schema = void (*)(Tensor, Tensor, Tensor, double);

    static void execute(Tensor out, Tensor x1, Tensor x2, double p);

    static common::OpDispatcher<schema> &dispatcher();
};

/**
 * @brief 非原地（Out-of-place）接口
 * @return 返回形状为 (M, N) 的新 Tensor
 */
Tensor cdist(Tensor x1, Tensor x2, double p = 2.0);

/**
 * @brief 显式指定输出接口
 */
void cdist_(Tensor out, Tensor x1, Tensor x2, double p = 2.0);

} // namespace infinicore::op
