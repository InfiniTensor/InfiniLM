#pragma once

#include "../../layers/causal_lm_templates/hybrid_decoder_layer.hpp"
#include "../../layers/common_modules.hpp"
#include "../qwen3_next/qwen3_next_gated_deltanet.hpp"
#include "qwen3_5_attention.hpp"

namespace infinilm::models::qwen3_5 {

using Qwen35DecoderLayer = infinilm::layers::causal_lm_templates::HybridDecoderLayer<
    Qwen35Attention,
    qwen3_next::Qwen3NextGatedDeltaNet,
    infinilm::layers::MLP>;

} // namespace infinilm::models::qwen3_5
