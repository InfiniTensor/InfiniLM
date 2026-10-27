#pragma once

#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "qwen3_5_model.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::qwen3_5 {

class Qwen35ForCausalLM
    : public infinilm::layers::causal_lm_templates::TextCausalLM<Qwen35Model> {
public:
    using Base = infinilm::layers::causal_lm_templates::TextCausalLM<Qwen35Model>;
    Qwen35ForCausalLM(infinilm::runtime::Runtime &runtime,
                      std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const engine::distributed::RankInfo &rank_info,
                      backends::AttentionBackend attention_backend,
                      const layers::moe::EPConfig &ep_config)
        : Base(runtime, model_config, rank_info, attention_backend, ep_config) {}

    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank_info) override {
        cache::validate_backend_cache_config(cache_config, attention_backend);
        if (cache_config != nullptr) {
            cache::validate_cache_runtime(forward_context.require_runtime(), rank_info);
        }
        if (cache_config == nullptr) {
            this->cache_config_.reset();
        } else {
            this->cache_config_ = cache_config->unique_copy();
        }
        this->model().reset_cache(cache_config, forward_context, attention_backend, rank_info);
    }
};

std::shared_ptr<infinilm::config::ModelConfig> prepare_qwen3_5_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_5
