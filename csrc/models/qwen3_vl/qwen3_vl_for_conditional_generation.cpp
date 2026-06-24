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
    auto text_config = std::make_shared<infinilm::config::ModelConfig>(text_config_json);
    const auto &dtype{model_config->get_dtype()};

    image_token_id_ = config_json.value("image_token_id", 151655);
    INFINICORE_NN_MODULE_INIT(language_model, text_config, device);
    INFINICORE_NN_MODULE_INIT(visual, config_json["vision_config"], dtype, device);
}

void Qwen3VLModel::replace_image_embeddings_(infinicore::Tensor inputs_embeds,
                                             const infinicore::Tensor &input_ids,
                                             const infinicore::Tensor &image_embeds) const {
    auto ids_cpu = input_ids->to(infinicore::Device::cpu());
    const int64_t *ids = reinterpret_cast<const int64_t *>(ids_cpu->data());
    size_t seq_len = ids_cpu->numel();
    size_t hidden_size = inputs_embeds->size(2);
    size_t image_idx = 0;
    for (size_t i = 0; i < seq_len; ++i) {
        if (static_cast<size_t>(ids[i]) != image_token_id_) {
            continue;
        }
        if (image_idx >= image_embeds->size(0)) {
            throw std::runtime_error("Qwen3VLModel: more image tokens than image embeddings");
        }
        inputs_embeds->narrow({{1, i, 1}})->copy_from(image_embeds->narrow({{0, image_idx, 1}})->view({1, 1, hidden_size}));
        image_idx++;
    }
    if (image_idx != image_embeds->size(0)) {
        throw std::runtime_error("Qwen3VLModel: image features and image tokens do not match");
    }
}

infinicore::Tensor Qwen3VLModel::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    if (input.pixel_values.has_value() && !input.pixel_values->empty()) {
        if (!input.tgt_sizes.has_value()) {
            throw std::runtime_error("Qwen3VLModel: image_grid_thw must be provided via tgt_sizes with pixel_values");
        }
        auto inputs_embeds = language_model_->embed_tokens(input_ids);
        auto input_offsets_cpu = input.input_offsets.value()->to(infinicore::Device::cpu());
        const int32_t *offsets = reinterpret_cast<const int32_t *>(input_offsets_cpu->data());
        auto req_ids = infinilm::global_state::get_forward_context().mm_metadata.image_req_ids;
        for (size_t i = 0; i < input.pixel_values->size(); ++i) {
            size_t req_id = req_ids.has_value() ? req_ids->at(i) : i;
            auto image_embeds = visual_->forward(input.pixel_values->at(i), input.tgt_sizes->at(i));
            size_t start = static_cast<size_t>(offsets[req_id]);
            size_t len = static_cast<size_t>(offsets[req_id + 1] - offsets[req_id]);
            auto embeds_slice = inputs_embeds->narrow({{1, start, len}});
            auto ids_slice = input_ids->narrow({{1, start, len}});
            replace_image_embeddings_(embeds_slice, ids_slice, image_embeds);
        }
        return language_model_->forward_embeds(inputs_embeds, input.position_ids.value());
    }
    return language_model_->forward(input);
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
    if (!config_json.contains("torch_dtype") || config_json["torch_dtype"].is_null()) {
        config_json["torch_dtype"] = text_config_json.value("dtype", "bfloat16");
    }
    if (!config_json.contains("dtype") || config_json["dtype"].is_null()) {
        config_json["dtype"] = text_config_json.value("dtype", "bfloat16");
    }
    if (!text_config_json.contains("torch_dtype") || text_config_json["torch_dtype"].is_null()) {
        text_config_json["torch_dtype"] = text_config_json.value("dtype", "bfloat16");
    }
    if (!text_config_json.contains("model_type") || text_config_json["model_type"] == "qwen3_vl_text") {
        text_config_json["model_type"] = "qwen3";
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
