#include "qwen3_5_for_causal_lm.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "../qwen3_next/qwen3_next_for_causal_lm.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_5 {

Qwen35ForCausalLM::Qwen35ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device) {
    model_config_ = model_config;
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype{model_config->get_dtype()};

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output Qwen35ForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void Qwen35ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        cache_config_.reset();
    } else {
        cache_config_ = cache_config->unique_copy();
    }
    model_->reset_cache(cache_config);
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_5" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3_5::create_qwen3_next_model_config: model_type is not qwen3_5");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (config_json.contains("text_config") && config_json["text_config"].is_object()) {
        const nlohmann::json &text_config_json = config_json["text_config"];
        for (auto it = text_config_json.begin(); it != text_config_json.end(); ++it) {
            if (!config_json.contains(it.key())) {
                config_json[it.key()] = it.value();
            }
        }
        if (!config_json.contains("dtype") && config_json.contains("torch_dtype")) {
            config_json["dtype"] = config_json["torch_dtype"];
        }
    }
    if (!config_json.contains("rope_theta") && config_json.contains("rope_parameters") && config_json["rope_parameters"].is_object() && config_json["rope_parameters"].contains("rope_theta")) {
        // TODO: This is only a temporary loader shim. Qwen3.6 uses mRoPE,
        // which needs proper support in InfiniCore instead of treating it as
        // plain RoPE through a top-level rope_theta.
        config_json["rope_theta"] = config_json["rope_parameters"]["rope_theta"];
    }
    if (!config_json.contains("partial_rotary_factor") && config_json.contains("rope_parameters") && config_json["rope_parameters"].is_object() && config_json["rope_parameters"].contains("partial_rotary_factor")) {
        config_json["partial_rotary_factor"] = config_json["rope_parameters"]["partial_rotary_factor"];
    }
    if (!config_json.contains("layer_types")) {
        size_t full_attention_interval = model_config->get<size_t>("full_attention_interval");
        size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        std::vector<std::string> layer_types;
        layer_types.reserve(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; i++) {
            layer_types.push_back(bool((i + 1) % full_attention_interval) ? "linear_attention" : "full_attention");
        }
        config_json["layer_types"] = layer_types;
    }

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    return model_config;
}

} // namespace infinilm::models::qwen3_5

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3_5,
    infinilm::models::qwen3_5::Qwen35ForCausalLM,
    infinilm::models::qwen3_5::create_qwen3_5_model_config);
} // namespace
