#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Vander {
public:
    // schema: output, input, N, increasing
    using schema = void (*)(Tensor, Tensor, int64_t, bool);

    static void execute(Tensor output, Tensor input, int64_t N, bool increasing);
    static common::OpDispatcher<schema> &dispatcher();
};

// N defaults to 0 (implying N = input.size(0), i.e., a square matrix)
Tensor vander(Tensor input, int64_t N = 0, bool increasing = false);
void vander_(Tensor output, Tensor input, int64_t N, bool increasing);

} // namespace infinicore::op
