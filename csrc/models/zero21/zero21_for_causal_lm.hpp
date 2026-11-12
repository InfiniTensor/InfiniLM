#pragma once

#include "../deepseek_v2/deepseek_v2_for_causal_lm.hpp"

#include <memory>
#include <string>

namespace infinilm::models::zero21 {

// Zero21ModelForCausalLM (model_type "zero21_model") shares the
// DeepseekV2 architecture: MLA attention without q_lora_rank,
// greedy-softmax MoE routing, and YaRN RoPE.
using Zero21ModelForCausalLM = infinilm::models::deepseek_v2::DeepseekV2ForCausalLM;

std::shared_ptr<infinilm::config::ModelConfig> create_zero21_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::zero21
