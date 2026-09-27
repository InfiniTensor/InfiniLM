#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Inner {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor out, Tensor input, Tensor other);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor inner(Tensor input, Tensor other);
void inner_(Tensor out, Tensor input, Tensor other);

} // namespace infinicore::op
