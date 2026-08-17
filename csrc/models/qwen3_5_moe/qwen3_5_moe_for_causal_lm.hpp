#pragma once

#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "../qwen3_5/qwen3_5_for_causal_lm.hpp"
#include "qwen3_5_moe_decoder_layer.hpp"

#include <memory>

namespace infinilm::models::qwen3_5_moe {

using Qwen35MoeLanguageModel = infinilm::layers::causal_lm_templates::TextModel<Qwen35MoeDecoderLayer>;
using Qwen35MoeModel = qwen3_5::Qwen35ModelTemplate<Qwen35MoeLanguageModel>;

class Qwen35MoeForConditionalGeneration
    : public infinilm::layers::causal_lm_templates::TextCausalLM<Qwen35MoeModel> {
public:
    using Base = infinilm::layers::causal_lm_templates::TextCausalLM<Qwen35MoeModel>;
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

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_moe_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_5_moe
