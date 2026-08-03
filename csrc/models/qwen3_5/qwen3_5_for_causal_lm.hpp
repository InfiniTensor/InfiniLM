#pragma once

#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "qwen3_5_model.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::qwen3_5 {

template <typename Model>
class Qwen35CausalLM : public infinilm::layers::causal_lm_templates::TextCausalLM<Model> {
public:
    using Base = infinilm::layers::causal_lm_templates::TextCausalLM<Model>;
    using Base::Base;

    void reset_cache(const cache::CacheConfig *cache_config) override {
        if (cache_config == nullptr) {
            this->cache_config_.reset();
        } else {
            this->cache_config_ = cache_config->unique_copy();
        }
        this->model().reset_cache(cache_config);
    }
};

using Qwen35ForCausalLM = Qwen35CausalLM<Qwen35Model>;

std::shared_ptr<infinilm::config::ModelConfig> prepare_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_5
