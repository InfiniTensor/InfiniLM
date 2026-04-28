#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class LogicalAnd {
public:
    // LogicalAnd 是二元操作，schema 通常定义为 (Output, Input1, Input2)
    using schema = void (*)(Tensor, Tensor, Tensor);

    static void execute(Tensor output, Tensor input1, Tensor input2);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor logical_and(Tensor input1, Tensor input2);
void logical_and_(Tensor output, Tensor input1, Tensor input2);

} // namespace infinicore::op
