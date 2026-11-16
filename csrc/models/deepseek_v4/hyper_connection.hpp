#pragma once

#include "runtime/tensor.hpp"

namespace infinilm::models::deepseek_v4 {

// Shapes are [tokens, streams, hidden] for residual and [tokens, hidden]
// for the collapsed activation. Mixing weights are retained in FP32.
struct HyperConnectionMix {
    runtime::Tensor activation;
    runtime::Tensor post;
    runtime::Tensor combination;
};

HyperConnectionMix hyper_connection_pre(
    runtime::Runtime &runtime, const runtime::Tensor &residual,
    const runtime::Tensor &projection, const runtime::Tensor &scale,
    const runtime::Tensor &bias, double norm_epsilon, double mix_epsilon,
    size_t sinkhorn_iterations);

runtime::Tensor hyper_connection_post(
    runtime::Runtime &runtime, const runtime::Tensor &activation,
    const runtime::Tensor &residual, const HyperConnectionMix &mix);

runtime::Tensor hyper_connection_head(
    runtime::Runtime &runtime, const runtime::Tensor &residual,
    const runtime::Tensor &projection, const runtime::Tensor &scale,
    const runtime::Tensor &bias, double norm_epsilon, double mix_epsilon);

} // namespace infinilm::models::deepseek_v4
