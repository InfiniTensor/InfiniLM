#pragma once

#include "../../layers/common_modules.hpp"
#include <memory>

namespace infinilm::models::mimo {

using MiMoMLP = infinilm::layers::MLP;

using MiMoAttention = infinilm::layers::attention::Attention;

using MiMoDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<MiMoAttention, MiMoMLP>;

using MiMoModel = infinilm::layers::causal_lm_templates::TextModel<MiMoDecoderLayer>;

using MiMoForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<MiMoModel>;

} // namespace infinilm::models::mimo

namespace infinilm::models::mimo {

std::shared_ptr<infinilm::config::ModelConfig> create_mimo_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::mimo
