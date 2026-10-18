#include "qwen3_vl_for_conditional_generation.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include <exception>
#include <stdexcept>
#include <string>

namespace infinilm::models::qwen3_vl {

Qwen3VLModel::Qwen3VLModel(infinilm::runtime::Runtime &runtime,
                           std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const engine::distributed::RankInfo &rank,
                           backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep) {
    models::validate_runtime_construction(model_config, runtime, rank);
    try {
        nlohmann::json &config_json = model_config->get_config_json();
        nlohmann::json &text_config_json = config_json["text_config"];
        std::shared_ptr<infinilm::config::ModelConfig> text_config = std::make_shared<infinilm::config::ModelConfig>(text_config_json);

        language_model_ = this->bind_child("language_model", std::make_shared<std::remove_reference_t<decltype(*language_model_)>>(runtime, text_config, rank, attention_backend, ep));

    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Qwen3VLModel::forward(
    const infinilm::InfinilmModel::Input &input,
    global_state::ForwardContext &forward_context) const {
    auto hidden_states = language_model_->forward(input, forward_context);
    return hidden_states;
}

Qwen3VLForConditionalGeneration::Qwen3VLForConditionalGeneration(infinilm::runtime::Runtime &runtime,
                                                                 std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                                 const engine::distributed::RankInfo &rank,
                                                                 backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep) {
    models::validate_runtime_construction(model_config, runtime, rank);
    try {
        model_config_ = model_config;
        const nlohmann::json &config_json = model_config->get_config_json();
        const nlohmann::json &text_config_json = config_json["text_config"];
        auto text_config = std::make_shared<infinilm::config::ModelConfig>(text_config_json);

        size_t hidden_size = text_config->get<size_t>("hidden_size");
        size_t vocab_size = text_config->get<size_t>("vocab_size");
        const auto &dtype{model_config->get_dtype()};

        model_ = this->bind_child("model", std::make_shared<std::remove_reference_t<decltype(*model_)>>(runtime, model_config, rank, attention_backend, ep));
        lm_head_ = this->bind_child("lm_head", std::make_shared<std::remove_reference_t<decltype(*lm_head_)>>(runtime, hidden_size, vocab_size, false, dtype));

    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::InfinilmModel::Output Qwen3VLForConditionalGeneration::forward(
    const infinilm::InfinilmModel::Input &input,
    global_state::ForwardContext &forward_context) const {
    auto hidden_states = model_->forward(input, forward_context);
    auto logits = lm_head_->forward(*forward_context.runtime, hidden_states);
    return {logits};
}

void Qwen3VLForConditionalGeneration::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank) {
    cache::validate_backend_cache_config(cache_config, attention_backend);
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr, forward_context, attention_backend, rank);
        return;
    }
    auto &runtime = forward_context.require_runtime();
    cache::validate_cache_runtime(runtime, rank);
    cache_config_ = cache_config->unique_copy();

    const nlohmann::json &config_json = model_config_->get_config_json();
    const nlohmann::json &text_config_json = config_json["text_config"];
    auto text_model_config = std::make_shared<infinilm::config::ModelConfig>(text_config_json);

    auto &kv_cache_vec = forward_context.kv_cache_vec;
    kv_cache_vec.clear();
    kv_cache_vec = std::move(default_allocate_kv_cache_tensors(runtime, rank, cache_config, text_model_config, attention_backend));
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_vl_text_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (!model_config || model_config->get<std::string>("model_type") != "qwen3_vl_text") {
        throw std::invalid_argument("Qwen3VL requires its real qwen3_vl_text configuration");
    }
    auto &text = model_config->get_config_json();
    if (model_config->get<int64_t>("num_attention_heads") <= 0) {
        throw std::invalid_argument("Qwen3VL text requires positive attention heads");
    }
    if (!text.contains("head_dim")) {
        text["head_dim"] = model_config->get<size_t>("hidden_size") / model_config->get<size_t>("num_attention_heads");
    }
    models::validate_dense_construction_metadata(model_config, "Qwen3VL text");
    return model_config;
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("qwen3_vl" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3_vl::create_qwen3_vl_model_config: model_type is not qwen3_vl");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    nlohmann::json &text_config_json = config_json["text_config"];
    auto text_config = std::make_shared<infinilm::config::ModelConfig>(text_config_json);
    const auto type = text_config->get<std::string>("model_type");
    if (type == "qwen3_vl_text") {
        create_qwen3_vl_text_model_config(text_config);
    } else if (type == "qwen3") {
        qwen3::create_qwen3_model_config(text_config);
    } else {
        throw std::invalid_argument("Unsupported real Qwen3VL text model_type");
    }
    text_config_json = text_config->get_config_json();
    if (!config_json.contains("torch_dtype")) {
        std::string dtype = text_config_json["dtype"];
        config_json["torch_dtype"] = dtype;
    }
    return model_config;
}

} // namespace infinilm::models::qwen3_vl

namespace {
// Config-only handler on the existing map; no second model creator.
const bool g_qwen3_vl_text_config = [] {
    infinilm::models::register_model_config("qwen3_vl_text",
                                            infinilm::models::qwen3_vl::create_qwen3_vl_text_model_config);
    return true;
}();
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    qwen3_vl,
    infinilm::models::qwen3_vl::Qwen3VLForConditionalGeneration,
    infinilm::models::qwen3_vl::create_qwen3_vl_model_config);
} // namespace
