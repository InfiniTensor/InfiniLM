#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Addcmul {
public:
    // schema: out, input, t1, t2, value
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, float);
    static void execute(Tensor out, Tensor input, Tensor t1, Tensor t2, float value);
    static common::OpDispatcher<schema> &dispatcher();
};
Tensor addcmul(Tensor input, Tensor t1, Tensor t2, float value);
void addcmul_(Tensor out, Tensor input, Tensor t1, Tensor t2, float value);
} // namespace infinicore::op
