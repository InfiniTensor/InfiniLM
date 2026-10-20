#pragma once

#include "../qwen3/qwen3_attention.hpp"
#include "qwen3_moe_sparse_moe_block.hpp"
#include <memory>

namespace infinilm::models::qwen3_moe {

using Qwen3MoeAttention = qwen3::Qwen3Attention;

using Qwen3MoeDecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<Qwen3MoeAttention, Qwen3MoeSparseMoeBlock>;

using Qwen3MoeModel = infinilm::layers::causal_lm_templates::TextModel<Qwen3MoeDecoderLayer, true>;

using Qwen3MoeForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<Qwen3MoeModel>;

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_moe

static_assert(std::is_constructible_v<infinilm::models::qwen3_moe::Qwen3MoeForCausalLM,
    infinilm::runtime::Runtime &, std::shared_ptr<infinilm::config::ModelConfig>,
    const infinilm::engine::distributed::RankInfo &, infinilm::backends::AttentionBackend,
    const infinilm::layers::moe::EPConfig &>, "Qwen3MoE selected constructor must consume explicit resources");
