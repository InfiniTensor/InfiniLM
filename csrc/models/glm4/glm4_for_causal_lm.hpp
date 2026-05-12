#pragma once

#include "glm4_decoder_layer.hpp"
#include <memory>

namespace infinilm::models::glm4 {

using Glm4Model = infinilm::layers::causal_lm_templates::TextModel<Glm4DecoderLayer>;

using Glm4ForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<Glm4Model>;

std::shared_ptr<infinilm::config::ModelConfig> create_glm4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::glm4

