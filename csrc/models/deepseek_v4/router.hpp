#pragma once

#include "runtime/tensor.hpp"

namespace infinilm::models::deepseek_v4 {

enum class RouterScore { SqrtSoftplus, Sigmoid, Softmax };

struct RoutedExperts {
    runtime::Tensor weights; // FP32 [tokens, top_k]
    runtime::Tensor indices; // INT64 [tokens, top_k]
};

// The optional correction bias affects ranking only. A hash layer supplies
// token IDs and the checkpoint's token-to-expert table instead of that bias.
RoutedExperts route_experts(
    runtime::Runtime &runtime, const runtime::Tensor &logits, size_t top_k,
    double route_scale, RouterScore score,
    const runtime::Tensor &correction_bias = {},
    const runtime::Tensor &token_ids = {}, const runtime::Tensor &token_to_expert = {});

} // namespace infinilm::models::deepseek_v4
