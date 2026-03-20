#pragma once
#include "../../layers/common_modules.hpp"
#include <memory>

namespace infinilm::models::qwen3_vl {

// /** @brief Type alias for Qwen3 MLP module */
using Qwen3VLTextMLP = infinilm::layers::MLP;

// /** @brief Qwen3 attention: only one of ENABLE_*_ATTN may be defined. Sentinel detects multiple assignment. */
using Qwen3VLTextAttention = infinilm::layers::Attention;

// /** @brief Qwen3 decoder layer type alias */
using Qwen3VLTextDecoderLayer = infinilm::layers::TextDecoderLayer<Qwen3VLTextAttention, Qwen3VLTextMLP>;

// /** @brief Qwen3 model architecture (without language modeling head) */
using Qwen3VLTextModel = infinilm::layers::TextModel<Qwen3VLTextDecoderLayer>;

} // namespace infinilm::models::qwen3_vl