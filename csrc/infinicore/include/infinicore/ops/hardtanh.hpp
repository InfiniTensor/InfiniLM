#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class HardTanh {
public:
    using schema = void (*)(Tensor, Tensor, float, float);
    static void execute(Tensor output, Tensor input, float min_val, float max_val);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor hardtanh(Tensor input, float min_val = -1.0f, float max_val = 1.0f);
void hardtanh_(Tensor output, Tensor input, float min_val = -1.0f, float max_val = 1.0f);

} // namespace infinicore::op
