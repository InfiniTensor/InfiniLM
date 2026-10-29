#pragma once

#include "../qwen3_next/qwen3_next_gated_deltanet.hpp"

namespace infinilm::models::clover_moe {

class CloverGatedDeltaNet : public qwen3_next::Qwen3NextGatedDeltaNet {
public:
    CloverGatedDeltaNet(runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> config,
                        size_t layer, const engine::distributed::RankInfo &rank,
                        backends::AttentionBackend backend, const layers::moe::EPConfig &ep);
    runtime::Tensor forward(const runtime::Tensor &hidden,
                            const global_state::ForwardContext &context) const;
    void process_weights_after_loading(runtime::Runtime &runtime, bool pre_transpose) override;

private:
    weights::Weight right_weight_, logit_scale_, history_bias_, history_c_;
    runtime::Tensor conv_weight_;
    size_t tp_rank_, tp_size_, full_key_dim_, full_value_dim_;
};

} // namespace infinilm::models::clover_moe
