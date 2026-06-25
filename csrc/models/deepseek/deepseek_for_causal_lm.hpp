#pragma once

#include "../../layers/common_modules.hpp"
#include "deepseek_decoder_layer.hpp"
#include <memory>

namespace infinilm::models::deepseek {

using DeepseekModel = infinilm::layers::causal_lm_templates::TextModel<DeepseekDecoderLayer>;

using DeepseekForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<DeepseekModel>;

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::deepseek
