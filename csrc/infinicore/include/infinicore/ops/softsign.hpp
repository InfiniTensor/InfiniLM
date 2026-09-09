#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Softsign {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor y, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};
// 返回新 Tensor 的函数接口
Tensor softsign(Tensor x);
void softsign_(Tensor y, Tensor x);
} // namespace infinicore::op
