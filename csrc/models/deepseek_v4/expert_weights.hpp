#pragma once

#include "weights/weights.hpp"

namespace infinilm::models::deepseek_v4 {

// Own the rank-local contiguous packs while binding the checkpoint's
// individual w1/w2/w3 names to views with the appropriate TP shard axis.
class W8A8ExpertWeights final : public weights::ModelWeights {
public:
    W8A8ExpertWeights(runtime::Runtime &runtime, size_t experts, size_t hidden_size,
                      size_t intermediate_size, size_t tp_rank, size_t tp_size, bool shared = false);

    const runtime::Tensor &gate_up() const noexcept { return gate_up_; }
    const runtime::Tensor &gate_up_scale() const noexcept { return gate_up_scale_; }
    const runtime::Tensor &down() const noexcept { return down_; }
    const runtime::Tensor &down_scale() const noexcept { return down_scale_; }

private:
    runtime::Tensor gate_up_;
    runtime::Tensor gate_up_scale_;
    runtime::Tensor down_;
    runtime::Tensor down_scale_;
};

} // namespace infinilm::models::deepseek_v4
