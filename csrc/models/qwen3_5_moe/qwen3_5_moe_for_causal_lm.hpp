#pragma once

#include "../qwen3_5/qwen3_5_for_causal_lm.hpp"
#include "qwen3_5_moe_model.hpp"

#include <memory>

namespace infinilm::models::qwen3_5_moe {

using Qwen35MoeForConditionalGeneration = qwen3_5::Qwen35CausalLM<Qwen35MoeModel>;

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_moe_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_5_moe
