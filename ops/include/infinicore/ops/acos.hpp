#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Acos {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor acos(Tensor input);
void acos_(Tensor output, Tensor input);
} // namespace infinicore::op
