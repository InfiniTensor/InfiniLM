#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class LogCumSumExp {
public:
    using schema = void (*)(Tensor, Tensor, int, bool, bool);

    static void execute(Tensor y, Tensor x, int axis, bool exclusive, bool reverse);

    static common::OpDispatcher<schema> &dispatcher();
};

Tensor logcumsumexp(Tensor x, int axis, bool exclusive = false, bool reverse = false);

void logcumsumexp_(Tensor y, Tensor x, int axis, bool exclusive = false, bool reverse = false);

} // namespace infinicore::op
