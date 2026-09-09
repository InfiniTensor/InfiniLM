#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Softplus {
public:
    // 修改 1: Schema 增加 float beta, float threshold
    using schema = void (*)(Tensor, Tensor, float, float);
    static void execute(Tensor y, Tensor x, float beta, float threshold);
    static common::OpDispatcher<schema> &dispatcher();
};
// default: beta = 1.0, threshold = 20.0
Tensor softplus(Tensor x, float beta = 1.0f, float threshold = 20.0f);

void softplus_(Tensor y, Tensor x, float beta = 1.0f, float threshold = 20.0f);

} // namespace infinicore::op
