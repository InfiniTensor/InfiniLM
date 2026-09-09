#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include "infinicore/tensor.hpp"

namespace infinicore::op {

class RandomSample {
public:
    using schema = void (*)(Tensor, Tensor, float, float, int, float);
    static void execute(Tensor indices, Tensor logits, float random_val, float topp, int topk, float temperature);
    static common::OpDispatcher<schema> &dispatcher();
};

// Out-of-place API
Tensor random_sample(Tensor logits, float random_val, float topp, int topk, float temperature);
// In-place API
void random_sample_(Tensor indices, Tensor logits, float random_val, float topp, int topk, float temperature);

} // namespace infinicore::op
