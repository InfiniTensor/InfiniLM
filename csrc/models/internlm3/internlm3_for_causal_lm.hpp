#pragma once

#include "../../layers/common_modules.hpp"
#include "../llama/llama_for_causal_lm.hpp"
#include <memory>

namespace infinilm::models::internlm3 {
using InternLM3ForCausalLM = infinilm::models::llama::LlamaForCausalLM;

std::shared_ptr<infinilm::config::ModelConfig> create_internlm3_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::internlm3
