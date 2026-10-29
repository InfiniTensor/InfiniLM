#pragma once

#include "../../global_state/forward_context.hpp"
#include "../qwen3_next/qwen3_next_sparse_moe_block.hpp"

namespace infinilm::models::clover_moe {

class CloverSparseMoe : public qwen3_next::Qwen3NextSparseMoeBlock {
public:
    CloverSparseMoe(runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> config,
                    size_t layer, const engine::distributed::RankInfo &rank,
                    backends::AttentionBackend backend, const layers::moe::EPConfig &ep);
    runtime::Tensor forward(runtime::Runtime &runtime, const runtime::Tensor &hidden,
                            const global_state::ForwardContext &context) const;

private:
    std::tuple<runtime::Tensor, runtime::Tensor> route(runtime::Runtime &runtime,
                                                       const runtime::Tensor &hidden, const global_state::ForwardContext &context) const;
    runtime::Tensor router_weight_;
    mutable runtime::Tensor cached_routed_;
    bool coreset_;
    double top_p_;
    size_t top_k_;
};

} // namespace infinilm::models::clover_moe
