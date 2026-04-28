#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <vector>

namespace infinicore::op {
class BroadcastTo {
public:
    // Schema: Output(y), Input(x)
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor y, Tensor x);
    static common::OpDispatcher<schema> &dispatcher();
};
Tensor broadcast_to(Tensor x, const std::vector<int64_t> &shape);
void broadcast_to_(Tensor y, Tensor x);

} // namespace infinicore::op
