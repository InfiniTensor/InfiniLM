#include "deepseek_v4_for_causal_lm.hpp"

#include "../models_registry.hpp"
#include "deepseek_v4_utils.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::deepseek_v4 {

DeepseekV4ForCausalLM::DeepseekV4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                             const infinicore::Device &device) {
    model_config_ = model_config;

    const auto &dtype = model_config->get_dtype();
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");

    model_ = this->register_module<DeepseekV4Model>("", model_config, device);
    INFINICORE_NN_MODULE_INIT(head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output DeepseekV4ForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto hidden_states = model_->forward(input_ids, positions);
    auto logits = head_->forward(hidden_states);
    return {logits};
}

void DeepseekV4ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    cached_input_ids_.clear();
    InfinilmModel::reset_cache(cache_config);
}

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("deepseek_v4" != model_type) {
        throw std::runtime_error("create_deepseek_v4_model_config: model_type is not deepseek_v4");
    }

    auto &config_json = model_config->get_config_json();
    if (!config_json.contains("dtype") && config_json.contains("torch_dtype")) {
        config_json["dtype"] = config_json["torch_dtype"];
    }
    if (!config_json.contains("quantization_config") && config_json.contains("compression_config")) {
        config_json["quantization_config"] = config_json["compression_config"];
    }
    if (!config_json.contains("num_key_value_heads")) {
        config_json["num_key_value_heads"] = 1;
    }
    if (config_json.contains("qk_rope_head_dim")) {
        const size_t head_dim = config_json.contains("head_dim")
                                    ? config_json.at("head_dim").get<size_t>()
                                    : config_json.at("hidden_size").get<size_t>() / config_json.at("num_attention_heads").get<size_t>();
        const size_t qk_rope_head_dim = config_json.at("qk_rope_head_dim").get<size_t>();
        config_json["head_dim"] = head_dim;
        config_json["partial_rotary_factor"] = static_cast<double>(qk_rope_head_dim) / static_cast<double>(head_dim);
    }

    // SGLang/vLLM DeepSeek-V4 uses GPT-J/interleaved RoPE (is_neox_style=false).
    model_config->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_J);

    return model_config;
}

} // namespace infinilm::models::deepseek_v4

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    deepseek_v4,
    infinilm::models::deepseek_v4::DeepseekV4ForCausalLM,
    infinilm::models::deepseek_v4::create_deepseek_v4_model_config);
} // namespace
