#pragma once

#include "../../layers/causal_lm_templates/hybrid_decoder_layer.hpp"
#include "../qwen3_5/qwen3_5_for_causal_lm.hpp"
#include "../qwen3_next/qwen3_next_gated_deltanet.hpp"
#include "../qwen3_next/qwen3_next_sparse_moe_block.hpp"

#include <memory>

namespace infinilm::models::qwen3_5_moe {

using Qwen35MoeDecoderLayer = infinilm::layers::causal_lm_templates::HybridDecoderLayer<
    qwen3_5::Qwen35Attention,
    qwen3_next::Qwen3NextGatedDeltaNet,
    qwen3_next::Qwen3NextSparseMoeBlock>;
using Qwen35MoeLanguageModel = infinilm::layers::causal_lm_templates::TextModel<Qwen35MoeDecoderLayer>;
using Qwen35MoeModel = qwen3_5::Qwen35ModelTemplate<Qwen35MoeLanguageModel>;
using Qwen35MoeForConditionalGeneration = qwen3_5::Qwen35CausalLM<Qwen35MoeModel>;

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_moe_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_5_moe
