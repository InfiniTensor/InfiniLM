#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Argwhere {
public:
    using schema = void (*)(void **, size_t *, Tensor);
    static void execute(void **, size_t *count, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};
Tensor argwhere(Tensor x);
} // namespace infinicore::op
