#pragma once

#include "../../layers/common_modules.hpp"
#include <memory>

namespace infinilm::models::mistral {

using MistralMLP = infinilm::layers::MLP;

using MistralAttention = infinilm::layers::attention::Attention;

using MistralDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<MistralAttention, MistralMLP>;

using MistralModel = infinilm::layers::causal_lm_templates::TextModel<MistralDecoderLayer, true>;

using MistralForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<MistralModel>;

} // namespace infinilm::models::mistral

namespace infinilm::models::mistral {

std::shared_ptr<infinilm::config::ModelConfig> create_mistral_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::mistral
