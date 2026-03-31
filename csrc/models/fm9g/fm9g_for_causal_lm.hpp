#pragma once

#include "../../layers/common_modules.hpp"
#include <memory>

namespace infinilm::models::fm9g {

using FM9GMLP = infinilm::layers::MLP;

using FM9GAttention = infinilm::layers::attention::Attention;

using FM9GDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<FM9GAttention, FM9GMLP>;

using FM9GModel = infinilm::layers::causal_lm_templates::TextModel<FM9GDecoderLayer>;

using FM9GForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<FM9GModel>;

} // namespace infinilm::models::fm9g

namespace infinilm::models::fm9g {

std::shared_ptr<infinilm::config::ModelConfig> create_fm9g_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::fm9g
