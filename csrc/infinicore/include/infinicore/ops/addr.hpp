#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Addr {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, float, float);
    static void execute(Tensor out, Tensor input, Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor addr(Tensor input, Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f);
void addr_(Tensor out, Tensor input, Tensor vec1, Tensor vec2, float beta = 1.0f, float alpha = 1.0f);
} // namespace infinicore::op
