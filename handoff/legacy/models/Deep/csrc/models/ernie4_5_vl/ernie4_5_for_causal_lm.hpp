#pragma once

#include "ernie4_5_model.hpp"

namespace infinilm::models::ernie4_5_vl {

class Ernie45ForConditionalGeneration : public InfinilmModel {
public:
    Ernie45ForConditionalGeneration(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);

    Output forward(const Input &input,
                   global_state::ForwardContext &forward_context) const override;

    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank) override;

protected:
    std::shared_ptr<Ernie45VisionModel> vision_model_;
    std::shared_ptr<Ernie45Model> model_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_ernie4_5_moe_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::ernie4_5_vl
