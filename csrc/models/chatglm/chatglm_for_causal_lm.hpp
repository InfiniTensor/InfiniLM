#pragma once

#include "../../layers/common_modules.hpp"
#include <memory>

namespace infinilm::models::chatglm {

using ChatglmMLP = infinilm::layers::MLP;

using ChatglmAttention = infinilm::layers::attention::Attention;

using ChatglmDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<ChatglmAttention, ChatglmMLP>;

using ChatglmModel = infinilm::layers::causal_lm_templates::TextModel<ChatglmDecoderLayer>;

using ChatglmForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<ChatglmModel>;

std::shared_ptr<infinilm::config::ModelConfig> create_chatglm_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::chatglm
