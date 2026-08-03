#pragma once

#include "../qwen3_5/qwen3_5_model.hpp"
#include "qwen3_5_moe_decoder_layer.hpp"

namespace infinilm::models::qwen3_5_moe {

using Qwen35MoeLanguageModel = infinilm::layers::causal_lm_templates::TextModel<Qwen35MoeDecoderLayer>;
using Qwen35MoeModel = qwen3_5::Qwen35ModelTemplate<Qwen35MoeLanguageModel>;

} // namespace infinilm::models::qwen3_5_moe
