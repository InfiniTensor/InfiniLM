#pragma once

#include "../qwen3/qwen3_attention.hpp"
#include "qwen3_moe_sparse_moe_block.hpp"

namespace infinilm::models::qwen3_moe {

using Qwen3MoeAttention = qwen3::Qwen3Attention;

using Qwen3MoeDecoderLayer = infinilm::layers::TextDecoderLayer<Qwen3MoeAttention, Qwen3MoeSparseMoeBlock>;

using Qwen3MoeModel = infinilm::layers::TextModel<Qwen3MoeDecoderLayer>;

using Qwen3MoeForCausalLM = infinilm::layers::TextCausalLM<Qwen3MoeModel>;

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);
} // namespace infinilm::models::qwen3_moe
