#pragma once

#include "../../layers/common_modules.hpp"
#include "qwen3_moe_sparse_moe_block.hpp"
namespace infinilm::models::qwen3_moe {

/** @brief Type alias for Qwen3 MoE attention module */
using StaticAttn = infinilm::models::layers::StaticAttention;
using PagedAttn = infinilm::models::layers::PagedAttention;
using FlashAttn = infinilm::models::layers::FlashAttention;

using Qwen3MoeAttention = std::variant<std::shared_ptr<StaticAttn>, std::shared_ptr<PagedAttn>, std::shared_ptr<FlashAttn>>;

/** @brief Qwen3 MoE decoder layer type alias */
using Qwen3MoeDecoderLayer = infinilm::models::layers::TemplateDecoderLayer<Qwen3MoeAttention, Qwen3MoeSparseMoeBlock>;

/** @brief Qwen3 MoE model architecture (without language modeling head) */
using Qwen3MoeModel = infinilm::models::layers::TemplateModel<Qwen3MoeDecoderLayer>;

/** @brief Qwen3 MoE model for Causal Language Modeling */
using Qwen3MoeForCausalLM = infinilm::models::layers::TemplateCausalLM<Qwen3MoeModel>;

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
