#pragma once

#include "qwen3_next_decoderLayer.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::qwen3_next {

using Qwen3NextModel = infinilm::layers::causal_lm_templates::TextModel<Qwen3NextDecoderLayer>;

class Qwen3NextForCausalLM : public InfinilmModel {
public:
    Qwen3NextForCausalLM(infinilm::runtime::Runtime &runtime,
                     std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const engine::distributed::RankInfo &rank_info,
                     backends::AttentionBackend attention_backend,
                     const layers::moe::EPConfig &ep_config);

    Output forward(const Input &input, global_state::ForwardContext &forward_context) const override;

    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank_info) override;

protected:
    std::shared_ptr<Qwen3NextModel> model_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_next_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_next
