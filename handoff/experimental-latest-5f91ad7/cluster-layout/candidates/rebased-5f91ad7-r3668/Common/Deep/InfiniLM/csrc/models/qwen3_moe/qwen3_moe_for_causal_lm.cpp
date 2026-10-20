#include "qwen3_moe_for_causal_lm.hpp"
#include "../models_registry.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace infinilm::models::qwen3_moe {

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_moe" != model_type) {
        throw std::runtime_error("create_qwen3_moe_model_config: model_type is not qwen3_moe");
    }
    if (model_config->get_quant_scheme() != infinilm::quantization::QuantScheme::NONE) {
        throw std::invalid_argument("Qwen3MoE requires NONE quantization");
    }
    const auto &json = model_config->get_config_json();
    if ((json.contains("decoder_sparse_step") && model_config->get<int64_t>("decoder_sparse_step") != 1)
        || (json.contains("mlp_only_layers") && (!json.at("mlp_only_layers").is_array() || !json.at("mlp_only_layers").empty()))) {
        throw std::invalid_argument("Qwen3MoE requires decoder_sparse_step=1 and empty mlp_only_layers");
    }
    for (const char *key : {"hidden_size", "vocab_size", "num_hidden_layers", "head_dim", "num_attention_heads", "num_key_value_heads", "num_experts", "num_experts_per_tok", "moe_intermediate_size"}) {
        if (model_config->get<int64_t>(key) <= 0) {
            throw std::invalid_argument("Qwen3MoE Factory requires positive dimensions");
        }
    }
    const size_t experts = model_config->get<size_t>("num_experts");
    const size_t topk = model_config->get<size_t>("num_experts_per_tok");
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t mid = model_config->get<size_t>("moe_intermediate_size");
    const size_t heads = model_config->get<size_t>("num_attention_heads");
    const size_t kv_heads = model_config->get<size_t>("num_key_value_heads");
    const size_t head_dim = model_config->get<size_t>("head_dim");
    const size_t index_limit = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    if (topk > experts || heads % kv_heads != 0
        || hidden > index_limit || mid > index_limit / 2
        || heads > index_limit / head_dim
        || hidden > std::numeric_limits<size_t>::max() / experts / 2 / mid / sizeof(uint16_t)) {
        throw std::invalid_argument("Qwen3MoE Factory configuration geometry is invalid");
    }
    if (json.contains("moe_align_block_size")) {
        const auto block = model_config->get<int64_t>("moe_align_block_size");
        if (block <= 0 || block > std::numeric_limits<int32_t>::max()) {
            throw std::invalid_argument("Qwen3MoE requires positive int32 moe_align_block_size");
        }
    }
    (void)model_config->get<bool>("norm_topk_prob");
    const double epsilon = model_config->get<double>("rms_norm_eps");
    if (!std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("Qwen3MoE Factory requires finite positive rms_norm_eps");
    }
    return model_config;
}

} // namespace infinilm::models::qwen3_moe

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    qwen3_moe,
    infinilm::models::qwen3_moe::Qwen3MoeForCausalLM,
    infinilm::models::qwen3_moe::create_qwen3_moe_model_config);
} // namespace
