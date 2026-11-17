#pragma once

#include "../../layers/causal_lm_templates/piecewise_text_causal_lm.hpp"
#include "qwen3_5_model.hpp"
#include <memory>

namespace infinilm::models::qwen3_5 {

class Qwen35ForCausalLM
    : public infinilm::layers::causal_lm_templates::PiecewiseTextCausalLM<Qwen35Model> {
public:
    using Base = infinilm::layers::causal_lm_templates::PiecewiseTextCausalLM<Qwen35Model>;
    using Base::Base;

    void reset_cache(const cache::CacheConfig *cache_config) override;
};

std::shared_ptr<infinilm::config::ModelConfig> prepare_qwen3_5_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_5
