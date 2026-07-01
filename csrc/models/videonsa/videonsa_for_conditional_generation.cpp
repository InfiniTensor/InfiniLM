#include "videonsa_for_conditional_generation.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "infinicore/ops/cat.hpp"
#include <stdexcept>
#include <string>

namespace infinilm::models::videonsa {

namespace {

std::shared_ptr<infinilm::config::ModelConfig> text_config_from(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    nlohmann::json &config_json = model_config->get_config_json();
    nlohmann::json text_config_json = config_json.contains("text_config") ? config_json["text_config"] : config_json;
    text_config_json["model_type"] = "videonsa";
    if (!text_config_json.contains("torch_dtype") && config_json.contains("torch_dtype")) {
        text_config_json["torch_dtype"] = config_json["torch_dtype"];
    }
    if (!text_config_json.contains("head_dim")) {
        text_config_json["head_dim"] = text_config_json["hidden_size"].get<size_t>() / text_config_json["num_attention_heads"].get<size_t>();
    }
    if (!text_config_json.contains("attention_bias")) {
        text_config_json["attention_bias"] = true;
    }
    return std::make_shared<infinilm::config::ModelConfig>(text_config_json);
}

} // namespace

VideoNSAForConditionalGeneration::VideoNSAForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                                   const infinicore::Device &device) {
    model_config_ = model_config;
    auto text_config = text_config_from(model_config);
    const size_t hidden_size = text_config->get<size_t>("hidden_size");
    const size_t vocab_size = text_config->get<size_t>("vocab_size");
    const auto &dtype{text_config->get_dtype()};

    INFINICORE_NN_MODULE_INIT(model, text_config, device);
    INFINICORE_NN_MODULE_INIT(visual, model_config->get_config_json()["vision_config"], dtype, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

void VideoNSAForConditionalGeneration::replace_embeddings(infinicore::Tensor inputs_embeds,
                                                          const infinicore::Tensor &vision_hidden,
                                                          const infinicore::Tensor &image_bound) const {
    auto bounds_cpu = image_bound->to(infinicore::Device::cpu());
    auto out_slice = inputs_embeds->squeeze(0);
    auto bound_slice = bounds_cpu->squeeze(0);
    auto bound_count = bound_slice->size(0);
    size_t vision_offset = 0;
    for (size_t i = 0; i < bound_count; ++i) {
        auto bound = bound_slice->narrow({{0, i, 1}});
        auto bound_ptr = reinterpret_cast<const int64_t *>(bound->data());
        auto start = bound_ptr[0];
        auto end = bound_ptr[1];
        if (end <= start) {
            continue;
        }
        const size_t len = static_cast<size_t>(end - start);
        auto patch_embed = vision_hidden->narrow({{0, vision_offset, len}});
        out_slice->narrow({{0, size_t(start), len}})->copy_from(patch_embed);
        vision_offset += len;
    }
}

infinilm::InfinilmModel::Output VideoNSAForConditionalGeneration::forward(const infinilm::InfinilmModel::Input &input) const {
    if (input.pixel_values.has_value() && input.pixel_values.value().size() > 0) {
        if (!input.image_bound.has_value() || !input.tgt_sizes.has_value()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: image_bound and tgt_sizes must be provided with pixel_values");
        }
        auto input_ids = input.input_ids.value();
        auto inputs_embeds = model_->embed_tokens(input_ids);
        auto input_offsets_cpu = input.input_offsets.value()->to(infinicore::Device::cpu());
        int32_t *offsets = reinterpret_cast<int32_t *>(input_offsets_cpu->data());

        const auto &image_req_ids = global_state::get_forward_context().mm_metadata.image_req_ids.value();
        if (input.pixel_values->size() != image_req_ids.size() || input.image_bound->size() != image_req_ids.size() || input.tgt_sizes->size() != image_req_ids.size()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: multimodal tensor lists must match image_req_ids");
        }

        std::vector<infinicore::Tensor> pixel_tensors;
        std::vector<infinicore::Tensor> grid_tensors;
        pixel_tensors.reserve(image_req_ids.size());
        grid_tensors.reserve(image_req_ids.size());
        for (size_t media_idx = 0; media_idx < image_req_ids.size(); ++media_idx) {
            pixel_tensors.push_back(input.pixel_values.value().at(media_idx));
            grid_tensors.push_back(input.tgt_sizes.value().at(media_idx));
        }
        auto batched_pixels = pixel_tensors.size() == 1 ? pixel_tensors.front() : infinicore::op::cat(pixel_tensors, 0);
        auto batched_grids = grid_tensors.size() == 1 ? grid_tensors.front() : infinicore::op::cat(grid_tensors, 0);
        auto batched_vision_hidden = visual_->forward(batched_pixels, batched_grids);

        size_t vision_offset = 0;
        for (size_t media_idx = 0; media_idx < image_req_ids.size(); ++media_idx) {
            const size_t req_id = image_req_ids[media_idx];
            auto bounds_cpu = input.image_bound.value().at(media_idx)->to(infinicore::Device::cpu())->squeeze(0);
            auto bound_count = bounds_cpu->size(0);
            auto bounds = reinterpret_cast<const int64_t *>(bounds_cpu->data());
            size_t vision_len = 0;
            for (size_t i = 0; i < bound_count; ++i) {
                auto start = bounds[i * 2];
                auto end = bounds[i * 2 + 1];
                if (end > start) {
                    vision_len += static_cast<size_t>(end - start);
                }
            }

            auto vision_hidden = batched_vision_hidden->narrow({{0, vision_offset, vision_len}});
            auto req_embeds = inputs_embeds->narrow({{1, size_t(offsets[req_id]), size_t(offsets[req_id + 1] - offsets[req_id])}});
            replace_embeddings(req_embeds, vision_hidden, input.image_bound.value().at(media_idx));
            vision_offset += vision_len;
        }

        auto hidden_states = model_->forward_embeds(inputs_embeds, input.position_ids.value());
        auto logits = lm_head_->forward(hidden_states);
        return {logits};
    }

    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void VideoNSAForConditionalGeneration::reset_cache(const cache::CacheConfig *cache_config) {
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();

    auto text_config = text_config_from(model_config_);
    auto &kv_cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
    kv_cache_vec.clear();
    const backends::AttentionBackend attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    kv_cache_vec = std::move(default_allocate_kv_cache_tensors(cache_config, text_config, attention_backend));
}

std::shared_ptr<infinilm::config::ModelConfig> create_videonsa_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("videonsa" != model_type) {
        throw std::runtime_error("infinilm::models::videonsa::create_videonsa_model_config: model_type is not videonsa");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (config_json.contains("text_config")) {
        nlohmann::json &text_config_json = config_json["text_config"];
        if (!text_config_json.contains("head_dim")) {
            text_config_json["head_dim"] = text_config_json["hidden_size"].get<size_t>() / text_config_json["num_attention_heads"].get<size_t>();
        }
        if (!text_config_json.contains("attention_bias")) {
            text_config_json["attention_bias"] = true;
        }
        if (!config_json.contains("torch_dtype") && text_config_json.contains("torch_dtype")) {
            config_json["torch_dtype"] = text_config_json["torch_dtype"];
        }
    } else {
        if (!config_json.contains("head_dim")) {
            config_json["head_dim"] = model_config->get<size_t>("hidden_size") / model_config->get<size_t>("num_attention_heads");
        }
        if (!config_json.contains("attention_bias")) {
            config_json["attention_bias"] = true;
        }
    }
    return model_config;
}

} // namespace infinilm::models::videonsa

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    videonsa,
    infinilm::models::videonsa::VideoNSAForConditionalGeneration,
    infinilm::models::videonsa::create_videonsa_model_config);
} // namespace
