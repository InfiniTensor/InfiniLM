#pragma once

#include "../../layers/TemplateCausalLM.hpp"
#include "../../layers/TemplateDecoderLayer.hpp"
#include "../../layers/TemplateModel.hpp"
#include "../../layers/attention.hpp"
#include "../../layers/mlp.hpp"
#include "qwen3_moe_sparse_moe_block.hpp"

namespace infinilm::models::qwen3_moe {

/** @brief Type alias for Qwen3 MoE attention module */
using StaticAttn = infinilm::models::layers::StaticAttention;
using PagedAttn = infinilm::models::layers::PagedAttention;
using FlashAttn = infinilm::models::layers::FlashAttention;

using Qwen3MoeAttention = std::variant<std::shared_ptr<StaticAttn>, std::shared_ptr<PagedAttn>, std::shared_ptr<FlashAttn>>;

/** @brief Qwen3 MoE decoder layer type alias */
using Qwen3MoeDecoderLayer = infinilm::models::layers::TemplateDecoderLayer<Qwen3MoeAttention, Qwen3MoeSparseMoeBlock>;

/** @brief Qwen3 MoE model architecture (without language modeling head) */
using Qwen3MoeModel = infinilm::models::layers::TemplateModel<Qwen3MoeDecoderLayer>;

/** @brief Qwen3 MoE model for Causal Language Modeling */
using Qwen3MoeForCausalLM = infinilm::models::layers::TemplateCausalLM<Qwen3MoeModel>;

} // namespace infinilm::models::qwen3_moe
