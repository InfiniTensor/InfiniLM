#pragma once

#include "../../layers/TemplateCausalLM.hpp"
#include "../../layers/TemplateDecoderLayer.hpp"
#include "../../layers/TemplateModel.hpp"
#include "../../layers/attention.hpp"
#include "../../layers/mlp.hpp"

namespace infinilm::models::qwen3 {

/** @brief Type alias for Qwen3 MLP module */
using Qwen3MLP = infinilm::models::layers::MLP;

/** @brief Type alias for Qwen3 attention module */
using Qwen3Attention = infinilm::models::layers::LlamaAttention;

/** @brief Qwen3 decoder layer type alias */
using Qwen3DecoderLayer = infinilm::models::layers::TemplateDecoderLayer<Qwen3Attention, Qwen3MLP>;

/** @brief Qwen3 model architecture (without language modeling head) */
using Qwen3Model = infinilm::models::layers::TemplateModel<Qwen3DecoderLayer>;

/** @brief Qwen3 model for Causal Language Modeling */
using Qwen3ForCausalLM = infinilm::models::layers::TemplateCausalLM<Qwen3Model>;

} // namespace infinilm::models::qwen3
