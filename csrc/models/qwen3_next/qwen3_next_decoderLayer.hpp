#pragma once

#include "../../layers/causal_lm_templates/hybrid_decoder_layer.hpp"
#include "qwen3_next_attention.hpp"
#include "qwen3_next_gated_deltanet.hpp"
#include "qwen3_next_sparse_moe_block.hpp"

namespace infinilm::models::qwen3_next {

using Qwen3NextDecoderLayer = infinilm::layers::causal_lm_templates::HybridDecoderLayer<
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock>;

} // namespace infinilm::models::qwen3_next
