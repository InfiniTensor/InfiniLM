#include "qwen3_5_for_causal_lm.hpp"

#include "../models_registry.hpp"
#include "infinicore/ops/gemm.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_5 {

// TextModel diagnostic hooks are compiled into this Qwen3.5 translation unit.

Qwen35ForCausalLM::Qwen35ForCausalLM(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype = model_config->get_dtype();
    fp32_lm_head_output_ = model_config->get_config_json().value(
                               "lm_head_output_dtype", std::string())
                        == "float32";

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(
        lm_head, hidden_size, vocab_size, false, dtype, device);
}

InfinilmModel::Output Qwen35ForCausalLM::forward(
    const InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    const char *dump_dir = std::getenv("INFINILM_LAYER_DUMP_DIR");
    const char *dump_numel = std::getenv("INFINILM_LAYER_DUMP_NUMEL");
    if (dump_dir != nullptr && dump_dir[0] != '\0'
        && dump_numel != nullptr && dump_numel[0] != '\0'
        && hidden_states->numel()
               == std::strtoull(dump_numel, nullptr, 10)) {
        hidden_states->debug(
            std::string(dump_dir) + "/infini_result_norm.bin");
    }
    infinicore::Tensor logits;
    if (fp32_lm_head_output_) {
        auto hidden = hidden_states->is_contiguous()
                        ? hidden_states
                        : hidden_states->contiguous();
        const size_t ndim = hidden->ndim();
        auto output_shape = hidden->shape();
        output_shape[ndim - 1] = lm_head_->out_features();
        logits = infinicore::Tensor::empty(
            output_shape, infinicore::DataType::F32, hidden->device());
        size_t rows = 1;
        for (size_t i = 0; i + 1 < ndim; ++i) {
            rows *= hidden->shape()[i];
        }
        auto weight = lm_head_->weight()->contiguous();
        infinicore::op::gemm_(
            logits->view({rows, lm_head_->out_features()}),
            hidden->view({rows, lm_head_->in_features()}),
            weight->permute({1, 0}),
            1.0f,
            0.0f);
    } else {
        logits = lm_head_->forward(hidden_states);
    }
    return {logits, hidden_states};
}

void Qwen35ForCausalLM::reset_cache(
    const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        cache_config_.reset();
    } else {
        cache_config_ = cache_config->unique_copy();
    }
    model_->reset_cache(cache_config);
}

std::shared_ptr<infinilm::config::ModelConfig> prepare_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
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
    if (!config_json.contains("position_id_axes")) {
        size_t position_id_axes = 1;
        if (config_json.contains("rope_parameters")
            && config_json["rope_parameters"].is_object()) {
            const auto &rope_parameters = config_json["rope_parameters"];
            if (rope_parameters.contains("mrope_section")
                && rope_parameters["mrope_section"].is_array()
                && !rope_parameters["mrope_section"].empty()) {
                position_id_axes = rope_parameters["mrope_section"].size();
            }
        }
        config_json["position_id_axes"] = position_id_axes;
    }
    if (!config_json.contains("rope_theta") && config_json.contains("rope_parameters") && config_json["rope_parameters"].is_object() && config_json["rope_parameters"].contains("rope_theta")) {
        // Normalize the nested HuggingFace field for the Qwen3.5 attention module.
        config_json["rope_theta"] = config_json["rope_parameters"]["rope_theta"];
    }
    if (!config_json.contains("partial_rotary_factor") && config_json.contains("rope_parameters") && config_json["rope_parameters"].is_object() && config_json["rope_parameters"].contains("partial_rotary_factor")) {
        config_json["partial_rotary_factor"] = config_json["rope_parameters"]["partial_rotary_factor"];
    }
    if (!config_json.contains("layer_types")) {
        const size_t full_attention_interval = model_config->get<size_t>("full_attention_interval");
        if (full_attention_interval == 0) {
            throw std::runtime_error("Qwen3.5 full_attention_interval must be positive");
        }
        const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        std::vector<std::string> layer_types;
        layer_types.reserve(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            layer_types.push_back(
                (i + 1) % full_attention_interval == 0
                    ? "full_attention"
                    : "linear_attention");
        }
        config_json["layer_types"] = std::move(layer_types);
    }
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    return model_config;
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_5" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3_5::create_qwen3_5_model_config: model_type is not qwen3_5");
    }
    return prepare_qwen3_5_model_config(model_config);
}

} // namespace infinilm::models::qwen3_5

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3_5,
    infinilm::models::qwen3_5::Qwen35ForCausalLM,
    infinilm::models::qwen3_5::create_qwen3_5_model_config);
} // namespace
