#pragma once

#include "runtime/tensor.hpp"

namespace infinilm::engine {

// One request composition for the rank worker.
void sample_request_token(
    infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor indices,
    infinilm::runtime::Tensor logits, float random_value,
    float top_p, int top_k, float temperature);

} // namespace infinilm::engine
