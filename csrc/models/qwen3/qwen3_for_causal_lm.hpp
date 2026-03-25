#pragma once

#include "../../layers/common_modules.hpp"
#include "qwen3_attention.hpp"

namespace infinilm::models::qwen3 {

using Qwen3MLP = infinilm::layers::MLP;

using Qwen3Attention = infinilm::models::qwen3::Qwen3Attention;

using Qwen3DecoderLayer = infinilm::layers::TextDecoderLayer<Qwen3Attention, Qwen3MLP>;

using Qwen3Model = infinilm::layers::TextModel<Qwen3DecoderLayer>;

using Qwen3ForCausalLM = infinilm::layers::TextCausalLM<Qwen3Model>;

} // namespace infinilm::models::qwen3

namespace infinilm::models::qwen3 {

static std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {

    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("qwen3" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3::create_qwen3_model_config: model_type is not qwen3");
    }

    return model_config;
}

} // namespace infinilm::models::qwen3
