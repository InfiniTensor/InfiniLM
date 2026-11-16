#pragma once

#include "runtime/tensor.hpp"

namespace infinilm::models::deepseek_v4 {

struct RotaryFrequencyConfig {
    size_t rotary_dim;
    size_t positions;
    double theta;
    // Zero selects the unscaled sliding-window table.
    size_t original_context = 0;
    double factor = 1.0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
};

// FP32 table [positions, rotary_dim]. The native RotaryEmbedding provider
// expects cosine and sine halves; the reference V4 layout uses interleaved pairs.
// Supply the live cache capacity rather than the checkpoint's maximum context.
runtime::Tensor make_rotary_frequencies(runtime::Runtime &runtime,
                                        const RotaryFrequencyConfig &config,
                                        bool split_cos_sin = false);

} // namespace infinilm::models::deepseek_v4
