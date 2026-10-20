#pragma once

#include "../../layers/common_modules.hpp"
#include <memory>

namespace infinilm::models::llama {

using LlamaMLP = infinilm::layers::MLP;

using LlamaAttention = infinilm::layers::attention::Attention;

using LlamaDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<LlamaAttention, LlamaMLP>;

using LlamaModel = infinilm::layers::causal_lm_templates::TextModel<LlamaDecoderLayer, true>;

using LlamaForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<LlamaModel>;

} // namespace infinilm::models::llama

namespace infinilm::models::llama {

std::shared_ptr<infinilm::config::ModelConfig> create_llama_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::llama
