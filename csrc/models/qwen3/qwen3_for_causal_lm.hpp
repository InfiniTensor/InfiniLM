#pragma once

#include "../../layers/causal_lm_templates/piecewise_text_causal_lm.hpp"
#include "../../layers/common_modules.hpp"
#include "qwen3_attention.hpp"

namespace infinilm::models::qwen3 {

using Qwen3MLP = infinilm::layers::MLP;

using Qwen3Attention = infinilm::models::qwen3::Qwen3Attention;

using Qwen3DecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<Qwen3Attention, Qwen3MLP>;

using Qwen3Model = infinilm::layers::causal_lm_templates::TextModel<Qwen3DecoderLayer>;

using Qwen3ForCausalLM = infinilm::layers::causal_lm_templates::PiecewiseTextCausalLM<Qwen3Model>;

} // namespace infinilm::models::qwen3

namespace infinilm::models::qwen3 {

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3
