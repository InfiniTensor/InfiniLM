#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Asin {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor asin(Tensor input);
void asin_(Tensor output, Tensor input);

} // namespace infinicore::op
