#include "granitemoehybrid_for_causal_lm.hpp"
#include "granitemoehybrid_allocate_kv_cache_tensors.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::granitemoehybrid {

GraniteMoeHybridForCausalLM::GraniteMoeHybridForCausalLM(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype{model_config->get_dtype()};

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);

    const float logits_scaling = model_config->get_or<float>("logits_scaling", 1.0f);
    if (logits_scaling <= 0.0f) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::GraniteMoeHybridForCausalLM: "
            "logits_scaling must be greater than zero");
    }
    lm_head_->set_alpha(1.0f / logits_scaling);
}

infinilm::InfinilmModel::Output GraniteMoeHybridForCausalLM::forward(
    const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void GraniteMoeHybridForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    reset_runtime_state();
    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.conv_state_vec.clear();
    forward_context.ssm_state_vec.clear();
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();

    forward_context.kv_cache_vec.clear();
    const auto attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    auto allocated_cache = granitemoehybrid_allocate_cache_tensors(
        cache_config,
        model_config_,
        attention_backend);
    forward_context.kv_cache_vec =
        std::move(allocated_cache.kv_cache_tensors);
    forward_context.conv_state_vec =
        std::move(allocated_cache.conv_state_tensors);
}

std::shared_ptr<infinilm::config::ModelConfig> create_granitemoehybrid_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("granitemoehybrid" != model_type) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::create_granitemoehybrid_model_config: "
            "model_type is not granitemoehybrid");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (!config_json.contains("layer_types")) {
        const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        config_json["layer_types"] = std::vector<std::string>(num_hidden_layers, "mamba");
    }

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }

    if (!config_json.contains("position_embedding_type") ||
        config_json.at("position_embedding_type").is_null()) {
        config_json["position_embedding_type"] = "nope";
    }
    const std::string position_embedding_type =
        model_config->get<std::string>("position_embedding_type");
    if ("rope" != position_embedding_type && "nope" != position_embedding_type) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::create_granitemoehybrid_model_config: "
            "position_embedding_type must be either 'rope' or 'nope'");
    }

    return model_config;
}

} // namespace infinilm::models::granitemoehybrid

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    granitemoehybrid,
    infinilm::models::granitemoehybrid::GraniteMoeHybridForCausalLM,
    infinilm::models::granitemoehybrid::create_granitemoehybrid_model_config);
} // namespace
