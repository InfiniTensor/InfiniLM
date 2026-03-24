#pragma once

#include "../../layers/common_modules.hpp"
#include "../qwen3/qwen3_attention.hpp"
#include "../qwen3_next/qwen3_next_sparse_moe_block.hpp"

namespace infinilm::models::qwen3_moe {

using Qwen3MoeAttention = qwen3::Qwen3Attention;
using Qwen3MoeSparseMoeBlock = qwen3_next::Qwen3NextSparseMoeBlock;

using Qwen3MoeDecoderLayer = infinilm::layers::TextDecoderLayer<Qwen3MoeAttention, Qwen3MoeSparseMoeBlock>;

using Qwen3MoeModel = infinilm::layers::TextModel<Qwen3MoeDecoderLayer>;

using Qwen3MoeForCausalLM = infinilm::layers::TextCausalLM<Qwen3MoeModel>;

} // namespace infinilm::models::qwen3_moe

namespace infinilm::models::qwen3_moe {
static std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_moe" != model_type) {
        throw std::runtime_error("create_qwen3_moe_model_config: model_type is not qwen3_moe");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (!config_json.contains("qk_norm")) {
        config_json["qk_norm"] = true;
    }

    return model_config;
}
} // namespace infinilm::models::qwen3_moe
