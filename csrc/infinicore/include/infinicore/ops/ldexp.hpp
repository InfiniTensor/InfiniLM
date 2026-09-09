#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Ldexp {
public:
    // Schema signature: output(out), input(x), other(exp)
    using schema = void (*)(Tensor, Tensor, Tensor);

    static void execute(Tensor output, Tensor input, Tensor other);
    static common::OpDispatcher<schema> &dispatcher();
};

// Functional API: Returns a new Tensor containing input * (2^other)
Tensor ldexp(Tensor input, Tensor other);

// In-place/Output-provided API
// Writes the result into 'output'
void ldexp_(Tensor output, Tensor input, Tensor other);

} // namespace infinicore::op
