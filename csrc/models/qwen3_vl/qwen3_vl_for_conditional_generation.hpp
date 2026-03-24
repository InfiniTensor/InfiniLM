#pragma once

#include "../../layers/common_modules.hpp"
#include "../../models/qwen3/qwen3_for_causal_lm.hpp"
#include <memory>

namespace infinilm::models::qwen3_vl {

using Qwen3VLTextModel = infinilm::models::qwen3::Qwen3Model;

class Qwen3VLModel : public infinicore::nn::Module {
public:
    Qwen3VLModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device) {

        // Initialize module
        nlohmann::json &config_json = model_config->get_config_json();
        nlohmann::json &text_config_json = config_json["text_config"];
        std::shared_ptr<infinilm::config::ModelConfig> text_config = std::make_shared<infinilm::config::ModelConfig>(text_config_json);

        INFINICORE_NN_MODULE_INIT(language_model, text_config, device);
    }

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const {
        auto hidden_states = language_model_->forward(input);
        return hidden_states;
    }

protected:
    INFINICORE_NN_MODULE(Qwen3VLTextModel, language_model);
};

class Qwen3VLForConditionalGeneration : public InfinilmModel {
public:
    Qwen3VLForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                    const infinicore::Device &device,
                                    engine::distributed::RankInfo rank_info,
                                    backends::AttentionBackend attention_backend = backends::AttentionBackend::Default)
        : Qwen3VLForConditionalGeneration(model_config, device) {}

    Qwen3VLForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                    const infinicore::Device &device) {

        const nlohmann::json &config_json = model_config->get_config_json();
        const nlohmann::json &text_config_json = config_json["text_config"];
        const auto &text_config = std::make_shared<infinilm::config::ModelConfig>(text_config_json);

        size_t hidden_size = text_config->get<size_t>("hidden_size");
        size_t vocab_size = text_config->get<size_t>("vocab_size");

        const auto &dtype{model_config->get_dtype()};

        // Initialize base model
        INFINICORE_NN_MODULE_INIT(model, model_config, device);
        INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
    }

    Output forward(const Input &input) const override {
        // 1. Forward through base model to get hidden states
        auto hidden_states = model_->forward(input);

        // 2. Apply language modeling head to get logits
        auto logits = lm_head_->forward(hidden_states);

        return {logits};
    }

    void reset_cache(const cache::CacheConfig *cache_config) override {
        if (cache_config == nullptr) {
            InfinilmModel::reset_cache(nullptr);
            return;
        }
        auto &model_config = infinilm::config::get_current_infinilm_config().model_config;
        const nlohmann::json &config_json = model_config->get_config_json();
        const nlohmann::json &text_config_json = config_json["text_config"];
        const auto &text_model_config = std::make_shared<infinilm::config::ModelConfig>(text_config_json);

        initialize_kv_cache(cache_config, text_model_config);
    }

protected:
    INFINICORE_NN_MODULE(Qwen3VLModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};
} // namespace infinilm::models::qwen3_vl

namespace infinilm::models::qwen3_vl {

static std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
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

    if (!text_config_json.contains("qk_norm")) {
        text_config_json["qk_norm"] = true;
    }

    return model_config;
}

} // namespace infinilm::models::qwen3_vl
