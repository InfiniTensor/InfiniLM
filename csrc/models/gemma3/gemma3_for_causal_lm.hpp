#pragma once

#include "../../layers/common_modules.hpp"
#include "gemma3_decoder_layer.hpp"
#include <memory>

namespace infinilm::models::gemma3 {

using Gemma3Model = infinilm::layers::causal_lm_templates::TextModel<Gemma3DecoderLayer>;

// Gemma-3 has no logit soft-capping, so the generic causal-LM template applies
// as-is (unlike Gemma-2, which needs a custom top level for final soft-cap).
using Gemma3ForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<Gemma3Model>;

} // namespace infinilm::models::gemma3

namespace infinilm::models::gemma3 {

std::shared_ptr<infinilm::config::ModelConfig> create_gemma3_text_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::gemma3
