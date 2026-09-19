#include "lfm2_for_causal_lm.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "lfm2_allocate_cache.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace infinilm::models::lfm2 {

Lfm2ForCausalLM::Lfm2ForCausalLM(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : Lfm2CausalLMBase(std::move(model_config), device) {}

void Lfm2ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    auto &forward_context = infinilm::global_state::get_forward_context();
    if (cache_config == nullptr) {
        cache_config_.reset();
        forward_context.kv_cache_vec.clear();
        forward_context.conv_state_vec.clear();
        forward_context.ssm_state_vec.clear();
        return;
    }

    cache_config_ = cache_config->unique_copy();
    const auto backend =
        infinilm::global_state::get_infinilm_config().attention_backend;
    auto allocated = allocate_lfm2_cache_tensors(
        cache_config, model_config_, backend);
    forward_context.kv_cache_vec = std::move(allocated.kv_cache_tensors);
    forward_context.conv_state_vec = std::move(allocated.conv_state_tensors);
    forward_context.ssm_state_vec.clear();
}

std::shared_ptr<infinilm::config::ModelConfig> create_lfm2_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (model_config->get<std::string>("model_type") != "lfm2") {
        throw std::runtime_error(
            "create_lfm2_model_config: model_type must be 'lfm2'");
    }

    auto &json = model_config->get_config_json();
    const size_t hidden_size = json.at("hidden_size").get<size_t>();
    const size_t num_heads = json.at("num_attention_heads").get<size_t>();
    if (hidden_size % num_heads != 0) {
        throw std::runtime_error(
            "create_lfm2_model_config: hidden_size must be divisible by num_attention_heads");
    }
    json["head_dim"] = hidden_size / num_heads;

    if (!json.contains("rms_norm_eps")) {
        json["rms_norm_eps"] = json.value(
            "norm_eps", json.value("block_norm_eps", 1e-5));
    }

    if (!json.contains("intermediate_size")) {
        size_t intermediate = json.at("block_ff_dim").get<size_t>();
        if (json.value("block_auto_adjust_ff_dim", false)) {
            intermediate = static_cast<size_t>(2.0 * intermediate / 3.0);
            intermediate = static_cast<size_t>(
                json.value("block_ffn_dim_multiplier", 1.0)
                * static_cast<double>(intermediate));
            const size_t multiple = json.value("block_multiple_of", 1UL);
            intermediate = multiple * ((intermediate + multiple - 1) / multiple);
        }
        json["intermediate_size"] = intermediate;
    }

    if (!json.contains("layer_types")) {
        const size_t num_layers = json.at("num_hidden_layers").get<size_t>();
        std::unordered_set<size_t> attention_layers;
        for (const auto &idx : json.at("full_attn_idxs")) {
            attention_layers.insert(idx.get<size_t>());
        }
        std::vector<std::string> layer_types;
        layer_types.reserve(num_layers);
        for (size_t i = 0; i < num_layers; ++i) {
            layer_types.push_back(
                attention_layers.count(i) ? "full_attention" : "short_conv");
        }
        json["layer_types"] = layer_types;
    }

    if (!json.contains("rope_theta") && json.contains("rope_parameters")) {
        json["rope_theta"] = json.at("rope_parameters").value(
            "rope_theta", 10000.0);
    }
    json["attention_bias"] = json.value("attention_bias", false);
    json["attention_output_bias"] =
        json.value("attention_output_bias", false);
    json["mlp_bias"] = json.value("mlp_bias", false);
    return model_config;
}

} // namespace infinilm::models::lfm2

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    lfm2,
    infinilm::models::lfm2::Lfm2ForCausalLM,
    infinilm::models::lfm2::create_lfm2_model_config);
} // namespace
