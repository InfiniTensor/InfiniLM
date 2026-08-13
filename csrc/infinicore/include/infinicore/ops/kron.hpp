#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Kron {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor output, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor kron(Tensor a, Tensor b);
void kron_(Tensor output, Tensor a, Tensor b);

} // namespace infinicore::op
