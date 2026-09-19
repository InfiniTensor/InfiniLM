#pragma once

#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "../../layers/causal_lm_templates/text_model.hpp"
#include "lfm2_decoder_layer.hpp"

#include <memory>

namespace infinilm::models::lfm2 {

using Lfm2Model =
    infinilm::layers::causal_lm_templates::TextModel<Lfm2DecoderLayer, Lfm2RMSNorm>;
using Lfm2CausalLMBase =
    infinilm::layers::causal_lm_templates::TextCausalLM<Lfm2Model>;

class Lfm2ForCausalLM : public Lfm2CausalLMBase {
public:
    Lfm2ForCausalLM(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    void reset_cache(const cache::CacheConfig *cache_config) override;
};

std::shared_ptr<infinilm::config::ModelConfig> create_lfm2_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::lfm2
