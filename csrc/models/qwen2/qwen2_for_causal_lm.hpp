#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::models::qwen2 {

using Qwen2MLP = infinilm::layers::MLP;

using Qwen2Attention = infinilm::layers::attention::Attention;

using Qwen2DecoderLayer = infinilm::layers::TextDecoderLayer<Qwen2Attention, Qwen2MLP>;

using Qwen2Model = infinilm::layers::TextModel<Qwen2DecoderLayer>;

using Qwen2ForCausalLM = infinilm::layers::TextCausalLM<Qwen2Model>;

} // namespace infinilm::models::qwen2

namespace infinilm::models::qwen2 {

std::shared_ptr<infinilm::config::ModelConfig> create_qwen2_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen2
