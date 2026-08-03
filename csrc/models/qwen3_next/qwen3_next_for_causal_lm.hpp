#pragma once

#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "qwen3_next_decoderLayer.hpp"

#include <memory>

namespace infinilm::models::qwen3_next {

using Qwen3NextModel = infinilm::layers::causal_lm_templates::TextModel<Qwen3NextDecoderLayer>;

class Qwen3NextForCausalLM
    : public infinilm::layers::causal_lm_templates::TextCausalLM<Qwen3NextModel> {
public:
    using Base = infinilm::layers::causal_lm_templates::TextCausalLM<Qwen3NextModel>;
    using Base::Base;

    void reset_cache(const cache::CacheConfig *cache_config) override;
};

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_next_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_next
