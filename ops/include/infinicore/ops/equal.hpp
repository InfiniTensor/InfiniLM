#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Equal {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);

    static void execute(Tensor out, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor equal(Tensor a, Tensor b);
void equal_(Tensor out, Tensor a, Tensor b);

} // namespace infinicore::op
