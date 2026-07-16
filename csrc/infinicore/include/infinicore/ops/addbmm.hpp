#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Addbmm {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, float, float);
    static void execute(Tensor output, Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha);

    static common::OpDispatcher<schema> &dispatcher();
};
Tensor addbmm(Tensor input, Tensor batch1, Tensor batch2, float beta = 1.0f, float alpha = 1.0f);

void addbmm_(Tensor output, Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha);

} // namespace infinicore::op
