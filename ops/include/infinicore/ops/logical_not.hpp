#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class LogicalNot {
public:
    // LogicalNot 是一元操作，schema 定义为 (Output, Input)
    using schema = void (*)(Tensor, Tensor);

    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

// 构造新 Tensor 返回结果
Tensor logical_not(Tensor input);

// 将结果写入指定的 output Tensor
void logical_not_(Tensor output, Tensor input);

} // namespace infinicore::op
