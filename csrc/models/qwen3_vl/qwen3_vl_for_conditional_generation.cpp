#include "qwen3_vl_for_conditional_generation.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::models::qwen3_vl {
namespace {

std::vector<int64_t> tensor_to_i64_vector(const infinicore::Tensor &tensor) {
    auto cpu = tensor->to(infinicore::Device::cpu());
    std::vector<int64_t> values(cpu->numel());
    if (cpu->dtype() == infinicore::DataType::I64) {
        const auto *data = reinterpret_cast<const int64_t *>(cpu->data());
        values.assign(data, data + cpu->numel());
        return values;
    }
    if (cpu->dtype() == infinicore::DataType::I32) {
        const auto *data = reinterpret_cast<const int32_t *>(cpu->data());
        for (size_t i = 0; i < cpu->numel(); ++i) {
            values[i] = static_cast<int64_t>(data[i]);
        }
        return values;
    }
    throw std::runtime_error(
        "Qwen3VLModel: metadata tensors must be int32 or int64");
}

std::shared_ptr<infinilm::config::ModelConfig> make_text_config(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    auto text_json = model_config->get_config_json().at("text_config");
    const auto &root = model_config->get_config_json();
    if (root.contains("condition_encoder_num_hidden_layers")) {
        text_json["num_hidden_layers"] = root.at("condition_encoder_num_hidden_layers");
    }
    if (root.contains("condition_encoder_skip_final_norm")) {
        text_json["condition_encoder_skip_final_norm"] = root.at("condition_encoder_skip_final_norm");
    }
    return std::make_shared<infinilm::config::ModelConfig>(std::move(text_json));
}

} // namespace

Qwen3VLModel::Qwen3VLModel(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    const auto &config = model_config->get_config_json();
    const auto &dtype = model_config->get_dtype();
    INFINICORE_NN_MODULE_INIT(language_model, make_text_config(model_config),
                              device);
    if (config.contains("vision_config") && !config.at("vision_config").is_null()) {
        INFINICORE_NN_MODULE_INIT(visual, config.at("vision_config"), dtype,
                                  device);
    }
}

infinicore::Tensor
Qwen3VLModel::forward(const infinilm::InfinilmModel::Input &input) const {
    if (!input.pixel_values.has_value() || input.pixel_values->empty()) {
        return language_model_->forward(input);
    }
    if (!visual_) {
        throw std::runtime_error("Qwen3VLModel: visual module is not initialized");
    }
    if (!input.image_grid_thw.has_value() || !input.image_bound.has_value() || !input.input_offsets.has_value()) {
        throw std::runtime_error("Qwen3VLModel: image_grid_thw, image_bound and "
                                 "input_offsets are required");
    }

    const auto &pixels = input.pixel_values.value();
    const auto &grids = input.image_grid_thw.value();
    const auto &bounds = input.image_bound.value();
    if (pixels.size() != grids.size() || pixels.size() != bounds.size()) {
        throw std::runtime_error(
            "Qwen3VLModel: multimodal input list sizes do not match");
    }

    const auto &request_ids = input.image_req_ids.has_value()
                                ? input.image_req_ids.value()
                                : global_state::get_forward_context()
                                      .mm_metadata.image_req_ids.value();
    if (request_ids.size() != pixels.size()) {
        throw std::runtime_error(
            "Qwen3VLModel: image_req_ids must match pixel_values");
    }

    const auto offsets = tensor_to_i64_vector(input.input_offsets.value());
    auto inputs_embeds = language_model_->embed_tokens(input.input_ids.value());
    std::vector<Qwen3VLDeepStackEmbedding> deepstack_embeddings;

    for (size_t image_idx = 0; image_idx < pixels.size(); ++image_idx) {
        const size_t request_id = request_ids[image_idx];
        if (request_id + 1 >= offsets.size()) {
            throw std::runtime_error(
                "Qwen3VLModel: image request id is out of range");
        }
        const auto bound = tensor_to_i64_vector(bounds[image_idx]);
        if (bound.empty() || bound.size() % 2 != 0) {
            throw std::runtime_error("Qwen3VLModel: invalid image_bound");
        }

        const size_t request_start = static_cast<size_t>(offsets[request_id]);
        const size_t request_length = static_cast<size_t>(offsets[request_id + 1] - offsets[request_id]);

        auto vision_output = visual_->forward_with_deepstack(pixels[image_idx], grids[image_idx]);
        for (const auto &features : vision_output.deepstack_features) {
            if (features->size(0) != vision_output.pooler_output->size(0)) {
                throw std::runtime_error(
                    "Qwen3VLModel: DeepStack and pooled visual features do not match");
            }
        }

        size_t visual_offset = 0;
        for (size_t range_idx = 0; range_idx < bound.size(); range_idx += 2) {
            if (bound[range_idx] < 0 || bound[range_idx + 1] < bound[range_idx]) {
                throw std::runtime_error("Qwen3VLModel: invalid image_bound range");
            }
            const size_t local_start = static_cast<size_t>(bound[range_idx]);
            const size_t local_end = static_cast<size_t>(bound[range_idx + 1]);
            if (local_end > request_length) {
                throw std::runtime_error(
                    "Qwen3VLModel: image_bound is outside its request");
            }
            const size_t visual_tokens = local_end - local_start;
            if (visual_offset + visual_tokens > vision_output.pooler_output->size(0)) {
                throw std::runtime_error("Qwen3VLModel: visual placeholder count "
                                         "exceeds visual output");
            }
            const size_t token_offset = request_start + local_start;
            inputs_embeds->narrow({{1, token_offset, visual_tokens}})
                ->copy_from(vision_output.pooler_output
                                ->narrow({{0, visual_offset, visual_tokens}})
                                ->unsqueeze(0));

            for (size_t deepstack_idx = 0;
                 deepstack_idx < vision_output.deepstack_features.size();
                 ++deepstack_idx) {
                deepstack_embeddings.push_back(
                    {vision_output.deepstack_features[deepstack_idx]->narrow(
                         {{0, visual_offset, visual_tokens}}),
                     token_offset, deepstack_idx});
            }
            visual_offset += visual_tokens;
        }
        if (visual_offset != vision_output.pooler_output->size(0)) {
            throw std::runtime_error(
                "Qwen3VLModel: visual placeholder count does not match visual output");
        }
    }

    return language_model_->forward_embeds(
        inputs_embeds, input.position_ids.value(), deepstack_embeddings);
}

Qwen3VLForConditionalGeneration::Qwen3VLForConditionalGeneration(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : condition_encoder_mode_(model_config->get_config_json().value(
        "condition_encoder_mode", false)) {
    model_config_ = model_config;
    const auto text_config = make_text_config(model_config);
    const size_t hidden_size = text_config->get<size_t>("hidden_size");
    const size_t vocab_size = text_config->get<size_t>("vocab_size");
    const auto &dtype = model_config->get_dtype();

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    if (!condition_encoder_mode_) {
        INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype,
                                  device);
    }
}

infinilm::InfinilmModel::Output Qwen3VLForConditionalGeneration::forward(
    const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    if (condition_encoder_mode_) {
        return {hidden_states};
    }
    return {lm_head_->forward(hidden_states), hidden_states};
}

void Qwen3VLForConditionalGeneration::reset_cache(
    const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();
    auto &kv_cache = global_state::get_forward_context().kv_cache_vec;
    kv_cache.clear();
    kv_cache = default_allocate_kv_cache_tensors(
        cache_config, make_text_config(model_config_),
        global_state::get_infinilm_config().attention_backend);
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_vl_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (model_config->get<std::string>("model_type") != "qwen3_vl") {
        throw std::runtime_error(
            "create_qwen3_vl_model_config: model_type is not qwen3_vl");
    }
    auto &config = model_config->get_config_json();
    auto &text_config = config.at("text_config");
    if (!config.contains("torch_dtype") || config.at("torch_dtype").is_null()) {
        config["torch_dtype"] = text_config.at("dtype");
    }
    config["position_id_axes"] = 3;
    return model_config;
}

} // namespace infinilm::models::qwen3_vl

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3_vl, infinilm::models::qwen3_vl::Qwen3VLForConditionalGeneration,
    infinilm::models::qwen3_vl::create_qwen3_vl_model_config);
} // namespace
