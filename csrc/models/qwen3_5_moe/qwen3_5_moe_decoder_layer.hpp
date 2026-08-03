#pragma once

#include "../../layers/causal_lm_templates/hybrid_decoder_layer.hpp"
#include "../qwen3_5/qwen3_5_attention.hpp"
#include "../qwen3_next/qwen3_next_gated_deltanet.hpp"
#include "qwen3_5_moe_sparse_moe_block.hpp"

namespace infinilm::models::qwen3_5_moe {

using Qwen35MoeDecoderLayer = infinilm::layers::causal_lm_templates::HybridDecoderLayer<
    qwen3_5::Qwen35Attention,
    qwen3_next::Qwen3NextGatedDeltaNet,
    Qwen35MoeSparseMoeBlock>;

} // namespace infinilm::models::qwen3_5_moe
