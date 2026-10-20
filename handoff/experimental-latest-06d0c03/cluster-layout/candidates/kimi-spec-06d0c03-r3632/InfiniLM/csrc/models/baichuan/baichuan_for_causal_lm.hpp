#pragma once

#include "../llama/llama_for_causal_lm.hpp"
#include <memory>

namespace infinilm::models::baichuan {
using BaichuanForCausalLM = infinilm::models::llama::LlamaForCausalLM;

std::shared_ptr<infinilm::config::ModelConfig> create_baichuan_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::baichuan
