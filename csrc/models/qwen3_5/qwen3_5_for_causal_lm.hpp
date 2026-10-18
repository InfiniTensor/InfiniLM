#pragma once

#include "qwen3_5_model.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::qwen3_5 {

class Qwen35ForCausalLM : public InfinilmModel {
public:
    Qwen35ForCausalLM(infinilm::runtime::Runtime &runtime,
                      std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const engine::distributed::RankInfo &rank_info,
                      backends::AttentionBackend attention_backend,
                      const layers::moe::EPConfig &ep_config);

    Output forward(const Input &input,
                   global_state::ForwardContext &forward_context) const override;

    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank_info) override;

protected:
    std::shared_ptr<Qwen35Model> model_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig> prepare_qwen3_5_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_5
