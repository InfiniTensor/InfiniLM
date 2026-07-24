#pragma once

#include "common/op.hpp"

namespace infinicore::op {
class Ones {

public:
    using schema = void (*)(Tensor);
    static void execute(Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor ones();
void ones_(Tensor output);
} // namespace infinicore::op
