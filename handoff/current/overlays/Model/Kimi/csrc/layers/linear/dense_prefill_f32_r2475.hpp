#pragma once
#include "../../execution/submit.hpp"
#include <cstddef>
#include <cstdint>

namespace infinilm::layers::linear {
// Explicit opt-in numerical candidate. The predicate rejects graph recording
// before any candidate allocation or submission. Decode rows stay on baseline.
bool use_dense_prefill_f32_r2475(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &input,
    const infinilm::runtime::Tensor &weight,
    infinilm::runtime::DataType output_dtype);

// Matrix views [M,K] * [K,N] -> [M,N], all BF16. Returns the owned FP32
// temporary for independent diagnostics; production callers may discard it.
infinilm::runtime::Tensor dense_prefill_f32_r2475(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &input,
    const infinilm::runtime::Tensor &weight,
    const infinilm::runtime::Tensor &output, float alpha);
}
extern "C" int modern_dense_prefill_stats_r2475(
    int device, std::uint64_t *out, std::size_t count);
