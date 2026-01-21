#include "llava_model.hpp"

#include "infinicore/ops.hpp"

#include <algorithm>
#include <stdexcept>

namespace infinilm::models::llava {

LlavaVisionTower::LlavaVisionTower(const ClipVisionConfig &config,
                                   const infinicore::DataType &dtype,
                                   const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(vision_model, config, dtype, device);
}

infinicore::Tensor LlavaVisionTower::forward_features(const infinicore::Tensor &pixel_values,
                                                      int64_t feature_layer) const {
    return vision_model_->forward_features(pixel_values, feature_layer);
}

LlavaProjector::LlavaProjector(const LlavaConfig &config,
                               const infinicore::DataType &dtype,
                               const infinicore::Device &device)
    : activation_(config.projector_hidden_act) {
    const auto &vision = config.vision_config;
    INFINICORE_NN_MODULE_INIT(linear_1, vision.hidden_size, config.text_config.hidden_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear_2, config.text_config.hidden_size, config.text_config.hidden_size, true, dtype, device);
}

infinicore::Tensor LlavaProjector::forward(const infinicore::Tensor &hidden_states) const {
    auto x = linear_1_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    if (activation_ == "gelu") {
        x = infinicore::op::gelu(x);
    } else if (activation_ == "relu") {
        x = infinicore::op::relu(x);
    } else {
        throw std::runtime_error("LlavaProjector: unsupported activation " + activation_);
    }
    return linear_2_->forward(x);
}

LlavaForConditionalGeneration::LlavaForConditionalGeneration(const LlavaConfig &config,
                                                             const infinicore::Device &device,
                                                             engine::distributed::RankInfo rank_info)
    : config_(config), rank_info_(rank_info) {
    INFINICORE_NN_MODULE_INIT(language_model, config.text_config, device, rank_info);
    INFINICORE_NN_MODULE_INIT(vision_tower, config.vision_config, config.dtype, device);
    INFINICORE_NN_MODULE_INIT(multi_modal_projector, config, config.dtype, device);
}

infinicore::Tensor LlavaForConditionalGeneration::build_merged_embeddings(
    const infinicore::Tensor &input_ids,
    const infinicore::Tensor &inputs_embeds,
    const infinicore::Tensor &image_features,
    infinicore::Tensor &position_ids_out) const {

    auto input_ids_cpu = input_ids->to(infinicore::Device::cpu());
    auto batch_size = input_ids_cpu->size(0);
    auto seq_len = input_ids_cpu->size(1);
    auto image_seq_len = image_features->size(1);
    auto hidden_size = inputs_embeds->size(2);

    std::vector<size_t> new_lengths(batch_size, seq_len);
    const auto *ids_ptr = reinterpret_cast<const int64_t *>(input_ids_cpu->data());
    for (size_t b = 0; b < batch_size; ++b) {
        size_t img_count = 0;
        for (size_t t = 0; t < seq_len; ++t) {
            auto token = ids_ptr[b * seq_len + t];
            if (token == config_.image_token_index) {
                img_count++;
            }
        }
        if (img_count > 0) {
            new_lengths[b] = seq_len + img_count * (image_seq_len - 1);
        }
    }

    size_t max_len = *std::max_element(new_lengths.begin(), new_lengths.end());
    auto merged = infinicore::Tensor::zeros({batch_size, max_len, hidden_size}, inputs_embeds->dtype(), inputs_embeds->device());

    // Build position_ids on CPU
    auto position_ids_cpu = infinicore::Tensor::zeros({batch_size, max_len}, infinicore::DataType::I64, infinicore::Device::cpu());
    auto *pos_ptr = reinterpret_cast<int64_t *>(position_ids_cpu->data());

    for (size_t b = 0; b < batch_size; ++b) {
        size_t out_pos = 0;
        for (size_t t = 0; t < seq_len; ++t) {
            auto token = ids_ptr[b * seq_len + t];
            if (token == config_.image_token_index) {
                auto img_slice = image_features->narrow({{0, b, 1}});
                auto dst = merged->narrow({{0, b, 1}, {1, out_pos, image_seq_len}});
                dst->copy_from(img_slice);
                for (size_t i = 0; i < image_seq_len; ++i) {
                    pos_ptr[b * max_len + out_pos + i] = static_cast<int64_t>(out_pos + i);
                }
                out_pos += image_seq_len;
            } else {
                auto src = inputs_embeds->narrow({{0, b, 1}, {1, t, 1}});
                auto dst = merged->narrow({{0, b, 1}, {1, out_pos, 1}});
                dst->copy_from(src);
                pos_ptr[b * max_len + out_pos] = static_cast<int64_t>(out_pos);
                out_pos += 1;
            }
        }
    }

    position_ids_out = position_ids_cpu->to(inputs_embeds->device());
    return merged;
}

InfinilmModel::Output LlavaForConditionalGeneration::forward(const InfinilmModel::Input &input) const {
    if (!input.input_ids.has_value()) {
        throw std::runtime_error("LlavaForConditionalGeneration: input_ids is required");
    }
    auto input_ids = input.input_ids.value();

    if (input.pixel_values.has_value() && input_ids->size(1) > 1) {
        auto pixel_values = input.pixel_values.value();
        auto image_features = vision_tower_->forward_features(pixel_values, config_.vision_feature_layer);

        if (config_.vision_feature_select_strategy == "default") {
            if (image_features->size(1) <= 1) {
                throw std::runtime_error("LlavaForConditionalGeneration: vision features too short to drop CLS");
            }
            image_features = image_features->narrow({{1, 1, image_features->size(1) - 1}});
        } else if (config_.vision_feature_select_strategy != "full") {
            throw std::runtime_error("LlavaForConditionalGeneration: unknown vision_feature_select_strategy");
        }

        auto projected = multi_modal_projector_->forward(image_features);
        auto inputs_embeds = language_model_->model().embed_tokens(input_ids);

        infinicore::Tensor merged_position_ids;
        auto merged_embeds = build_merged_embeddings(input_ids, inputs_embeds, projected, merged_position_ids);

        auto hidden_states = language_model_->model().forward_embeds(
            merged_embeds,
            merged_position_ids,
            input.past_sequence_lengths,
            input.total_sequence_lengths,
            input.input_offsets,
            input.block_tables,
            input.slot_mapping);

        auto logits = language_model_->logits_from_hidden(hidden_states);
        return {logits};
    }

    return language_model_->forward(input);
}

void LlavaForConditionalGeneration::reset_cache(const cache::CacheConfig *cache_config) {
    language_model_->reset_cache(cache_config);
}

uint32_t LlavaForConditionalGeneration::compress_kv_cache_inplace(uint32_t seq_len,
                                                                   size_t batch_size,
                                                                   const cache::KVCompressionConfig &cfg) {
    return language_model_->compress_kv_cache_inplace(seq_len, batch_size, cfg);
}

} // namespace infinilm::models::llava
