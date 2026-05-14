#pragma once

#include "../../layers/common_modules.hpp"
#include "../glm4/glm4_attention.hpp"
#include <memory>

namespace infinilm::models::chatglm {

using ChatglmMLP = infinilm::layers::MLP;

// Reuse Glm4Attention as ChatGLM and GLM4 share the identical attention layer
using ChatglmAttention = infinilm::models::glm4::Glm4Attention;

using ChatglmDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<ChatglmAttention, ChatglmMLP>;

using ChatglmModel = infinilm::layers::causal_lm_templates::TextModel<ChatglmDecoderLayer>;

using ChatglmForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<ChatglmModel>;

std::shared_ptr<infinilm::config::ModelConfig> create_chatglm_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::chatglm
