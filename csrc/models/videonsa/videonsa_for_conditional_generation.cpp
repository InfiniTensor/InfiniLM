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

size_t visual_length_from_grid(const infinicore::Tensor &grid_tensor,
                               size_t spatial_merge_size) {
    if (spatial_merge_size == 0) {
        throw std::runtime_error("VideoNSAForConditionalGeneration: spatial_merge_size must be positive");
    }
    auto grid_cpu = grid_tensor->to(infinicore::Device::cpu());
    auto rows = grid_cpu->size(0);
    auto grid_ptr = reinterpret_cast<const int64_t *>(grid_cpu->data());
    size_t total = 0;
    for (size_t i = 0; i < rows; ++i) {
        const int64_t grid_t_i = grid_ptr[i * 3];
        const int64_t grid_h_i = grid_ptr[i * 3 + 1];
        const int64_t grid_w_i = grid_ptr[i * 3 + 2];
        if (grid_t_i <= 0 || grid_h_i <= 0 || grid_w_i <= 0) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: invalid grid_thw");
        }
        if (grid_h_i % static_cast<int64_t>(spatial_merge_size) != 0
            || grid_w_i % static_cast<int64_t>(spatial_merge_size) != 0) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: grid_thw is not divisible by spatial_merge_size");
        }
        const size_t grid_t = static_cast<size_t>(grid_t_i);
        const size_t llm_grid_h = static_cast<size_t>(grid_h_i / static_cast<int64_t>(spatial_merge_size));
        const size_t llm_grid_w = static_cast<size_t>(grid_w_i / static_cast<int64_t>(spatial_merge_size));
        total += grid_t * llm_grid_h * llm_grid_w;
    }
    return total;
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
                                                          const infinicore::Tensor &image_bound,
                                                          const infinicore::Tensor &image_embed_bound) const {
    auto bounds_cpu = image_bound->to(infinicore::Device::cpu());
    auto embed_bounds_cpu = image_embed_bound->to(infinicore::Device::cpu());
    ASSERT_EQ(inputs_embeds->size(0), 1);
    ASSERT_EQ(bounds_cpu->size(0), 1);
    ASSERT_EQ(bounds_cpu->size(2), 2);
    ASSERT_EQ(embed_bounds_cpu->size(0), 1);
    ASSERT_EQ(embed_bounds_cpu->size(1), bounds_cpu->size(1));
    ASSERT_EQ(embed_bounds_cpu->size(2), 2);
    auto out_slice = inputs_embeds->squeeze(0);
    auto bound_slice = bounds_cpu->squeeze(0);
    auto embed_bound_slice = embed_bounds_cpu->squeeze(0);
    auto bound_count = bound_slice->size(0);
    for (size_t i = 0; i < bound_count; ++i) {
        auto bound = bound_slice->narrow({{0, i, 1}});
        auto bound_ptr = reinterpret_cast<const int64_t *>(bound->data());
        const int64_t start = bound_ptr[0];
        const int64_t end = bound_ptr[1];
        if (start < 0 || end < start) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: invalid image_bound");
        }
        if (start == end) {
            continue;
        }
        const size_t len = static_cast<size_t>(end - start);

        auto embed_bound = embed_bound_slice->narrow({{0, i, 1}});
        auto embed_bound_ptr = reinterpret_cast<const int64_t *>(embed_bound->data());
        const int64_t src_start_i = embed_bound_ptr[0];
        const int64_t src_end_i = embed_bound_ptr[1];
        if (src_start_i < 0 || src_end_i < src_start_i) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: invalid image_embed_bound");
        }
        if (src_end_i - src_start_i != static_cast<int64_t>(len)) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: image_bound and image_embed_bound length mismatch");
        }
        const size_t src_start = static_cast<size_t>(src_start_i);
        const size_t src_len = len;
        if (static_cast<size_t>(end) > out_slice->size(0)
            || src_start > vision_hidden->size(0)
            || src_len > vision_hidden->size(0) - src_start) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: multimodal embedding bounds are out of range");
        }
        auto patch_embed = vision_hidden->narrow({{0, src_start, src_len}});
        out_slice->narrow({{0, size_t(start), len}})->copy_from(patch_embed);
    }
}

infinilm::InfinilmModel::Output VideoNSAForConditionalGeneration::forward(const infinilm::InfinilmModel::Input &input) const {
    if (input.pixel_values.has_value() && input.pixel_values.value().size() > 0) {
        if (!input.input_ids.has_value()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: input_ids is required");
        }
        if (!input.image_bound.has_value() || !input.tgt_sizes.has_value()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: image_bound and tgt_sizes must be provided with pixel_values");
        }
        if (!input.input_offsets.has_value()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: input_offsets is required with pixel_values");
        }
        auto input_ids = input.input_ids.value();
        auto inputs_embeds = model_->embed_tokens(input_ids);
        auto input_offsets_cpu = input.input_offsets.value()->to(infinicore::Device::cpu());
        int32_t *offsets = reinterpret_cast<int32_t *>(input_offsets_cpu->data());

        const auto &mm_metadata = global_state::get_forward_context().mm_metadata;
        if (!mm_metadata.image_req_ids.has_value()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: image_req_ids must be provided with pixel_values");
        }
        const auto &image_req_ids = mm_metadata.image_req_ids.value();
        if (input.pixel_values->size() != image_req_ids.size() || input.image_bound->size() != image_req_ids.size() || input.tgt_sizes->size() != image_req_ids.size()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: multimodal tensor lists must match image_req_ids");
        }
        if (!input.image_embed_bound.has_value()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: image_embed_bound must be provided with pixel_values");
        }
        if (input.image_embed_bound->size() != image_req_ids.size()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: image_embed_bound must match image_req_ids");
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
        size_t spatial_merge_size = 2;
        const auto &vision_config = model_config_->get_config_json()["vision_config"];
        if (vision_config.contains("spatial_merge_size")) {
            spatial_merge_size = vision_config["spatial_merge_size"].get<size_t>();
        }
        const size_t num_offsets = input_offsets_cpu->size(0);
        for (size_t media_idx = 0; media_idx < image_req_ids.size(); ++media_idx) {
            const size_t req_id = image_req_ids[media_idx];
            if (num_offsets < 2 || req_id >= num_offsets - 1) {
                throw std::runtime_error("VideoNSAForConditionalGeneration: image_req_ids is out of input_offsets range");
            }
            const int32_t req_start = offsets[req_id];
            const int32_t req_end = offsets[req_id + 1];
            if (req_start < 0 || req_end < req_start) {
                throw std::runtime_error("VideoNSAForConditionalGeneration: invalid input_offsets");
            }
            auto grid_tensor = input.tgt_sizes.value().at(media_idx);
            const size_t vision_len = visual_length_from_grid(grid_tensor, spatial_merge_size);
            if (vision_offset > batched_vision_hidden->size(0)
                || vision_len > batched_vision_hidden->size(0) - vision_offset) {
                throw std::runtime_error("VideoNSAForConditionalGeneration: visual hidden size does not match tgt_sizes");
            }

            auto vision_hidden = batched_vision_hidden->narrow({{0, vision_offset, vision_len}});
            auto req_embeds = inputs_embeds->narrow({{1, static_cast<size_t>(req_start), static_cast<size_t>(req_end - req_start)}});
            replace_embeddings(
                req_embeds,
                vision_hidden,
                input.image_bound.value().at(media_idx),
                input.image_embed_bound.value().at(media_idx));
            vision_offset += vision_len;
        }
        if (vision_offset != batched_vision_hidden->size(0)) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: unused visual hidden states after replacement");
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
