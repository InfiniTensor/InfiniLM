#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class LogSoftmax {
public:
    // Schema signature: output(out), input, dim
    using schema = void (*)(Tensor, Tensor, int64_t);

    static void execute(Tensor output, Tensor input, int64_t dim);
    static common::OpDispatcher<schema> &dispatcher();
};

// Functional API: Returns the result tensor
Tensor log_softmax(Tensor input, int64_t dim);

// In-place/Output-provided API
void log_softmax_(Tensor output, Tensor input, int64_t dim);

} // namespace infinicore::op
