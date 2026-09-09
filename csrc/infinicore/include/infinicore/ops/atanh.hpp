#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Atanh {
public:
    // schema 定义为：void(输出 Tensor, 输入 Tensor)
    using schema = void (*)(Tensor, Tensor);

    // 执行函数
    static void execute(Tensor y, Tensor a);

    // 获取算子分发器，用于多后端（CPU/CUDA 等）匹配
    static common::OpDispatcher<schema> &dispatcher();
};

/**
 * @brief 计算输入 Tensor 的反双曲正切值 (out-of-place)
 * @param a 输入 Tensor
 * @return 包含结果的新 Tensor
 */
Tensor atanh(Tensor a);

/**
 * @brief 计算输入 Tensor 的反双曲正切值 (in-place / specified output)
 * @param y 输出 Tensor
 * @param a 输入 Tensor
 */
void atanh_(Tensor y, Tensor a);

} // namespace infinicore::op
