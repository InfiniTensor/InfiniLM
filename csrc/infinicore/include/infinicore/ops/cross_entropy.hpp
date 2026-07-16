#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class CrossEntropy {
public:
    // Schema 定义：函数指针类型
    // CrossEntropy 需要接收三个 Tensor: Output (Loss), Input (Logits), Target (Labels)
    using schema = void (*)(Tensor, Tensor, Tensor);

    // 执行入口
    static void execute(Tensor output, Tensor input, Tensor target);

    // 分发器访问接口
    static common::OpDispatcher<schema> &dispatcher();
};

// ==================================================================
// 对外 Functional API
// ==================================================================

// 1. Out-of-place 接口：
// 输入 Logits 和 Target，内部自动创建 Output Tensor 并返回
Tensor cross_entropy(Tensor input, Tensor target);

// 2. Explicit Output 接口 (类似于 In-place 风格)：
// 用户显式提供 Output Tensor 用于存储结果
// 注意：虽然命名带有下划线 _，但通常 CrossEntropy 无法真正原地修改 input，
// 所以这里只是表示“写入指定的 output 内存”
void cross_entropy_(Tensor output, Tensor input, Tensor target);

} // namespace infinicore::op
