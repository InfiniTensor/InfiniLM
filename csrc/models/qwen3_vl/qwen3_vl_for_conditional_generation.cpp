#include "qwen3_vl_for_conditional_generation.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include <stdexcept>
#include <string>

namespace infinilm::models::qwen3_vl {

Qwen3VLModel::Qwen3VLModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device) {
    nlohmann::json &config_json = model_config->get_config_json();
    nlohmann::json &text_config_json = config_json["text_config"];
    std::shared_ptr<infinilm::config::ModelConfig> text_config = std::make_shared<infinilm::config::ModelConfig>(text_config_json);

    INFINICORE_NN_MODULE_INIT(language_model, text_config, device);
}

infinicore::Tensor Qwen3VLModel::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = language_model_->forward(input);
    return hidden_states;
}

Qwen3VLForConditionalGeneration::Qwen3VLForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                                 const infinicore::Device &device) {
    model_config_ = model_config;
    const nlohmann::json &config_json = model_config->get_config_json();
    const nlohmann::json &text_config_json = config_json["text_config"];
    auto text_config = std::make_shared<infinilm::config::ModelConfig>(text_config_json);

    size_t hidden_size = text_config->get<size_t>("hidden_size");
    size_t vocab_size = text_config->get<size_t>("vocab_size");
    const auto &dtype{model_config->get_dtype()};

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output Qwen3VLForConditionalGeneration::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void Qwen3VLForConditionalGeneration::reset_cache(const cache::CacheConfig *cache_config) {
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();

    const nlohmann::json &config_json = model_config_->get_config_json();
    const nlohmann::json &text_config_json = config_json["text_config"];
    auto text_model_config = std::make_shared<infinilm::config::ModelConfig>(text_config_json);

    auto &kv_cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
    kv_cache_vec.clear();
    const backends::AttentionBackend attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    kv_cache_vec = std::move(default_allocate_kv_cache_tensors(cache_config, text_model_config, attention_backend));
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("qwen3_vl" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3_vl::create_qwen3_vl_model_config: model_type is not qwen3_vl");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    nlohmann::json &text_config_json = config_json["text_config"];
    if (!config_json.contains("torch_dtype")) {
        std::string dtype = text_config_json["dtype"];
        config_json["torch_dtype"] = dtype;
    }
    return model_config;
}

} // namespace infinilm::models::qwen3_vl

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3_vl,
    infinilm::models::qwen3_vl::Qwen3VLForConditionalGeneration,
    infinilm::models::qwen3_vl::create_qwen3_vl_model_config);
} // namespace
