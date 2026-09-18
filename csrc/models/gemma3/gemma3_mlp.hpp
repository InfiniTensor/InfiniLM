#pragma once

#include "../gemma2/gemma2_mlp.hpp"

namespace infinilm::models::gemma3 {

// Gemma-3's MLP is identical to Gemma-2's (SwiGLU layout with
// gelu_pytorch_tanh), so reuse the implementation (qwen3_moe aliases qwen3 the
// same way).
using Gemma3MLP = infinilm::models::gemma2::Gemma2MLP;

} // namespace infinilm::models::gemma3
