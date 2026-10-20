#include "ernie4_5_model.hpp"
#include "../models_registry.hpp"

#include "../../global_state/forward_context.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <exception>

namespace infinilm::models::ernie4_5_vl {

Ernie45Model::Ernie45Model(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config)
    : model_config_(model_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info, attention_backend, ep_config);
    try {
    const auto &dtype = model_config->get_dtype();
    auto &config_json = model_config->get_config_json();
    if (config_json.contains("vision_config") && config_json["vision_config"].is_object()) {
            resampler_model_ = std::make_shared<std::remove_reference_t<decltype(*resampler_model_)>>(runtime, config_json, dtype);
            this->bind_child("resampler_model", resampler_model_);
    }
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
        mrope_cache_ = build_ernie45_mrope_cache(runtime, model_config, runtime.device());

        embed_tokens_ = std::make_shared<std::remove_reference_t<decltype(*embed_tokens_)>>(runtime, vocab_size, hidden_size, std::nullopt, dtype);
        this->bind_child("embed_tokens", embed_tokens_);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
            layers_.push_back(std::make_shared<Ernie45DecoderLayer>(runtime, model_config, i, mrope_cache_, rank_info, attention_backend, ep_config));
            this->bind_child("layers." + std::to_string(i), layers_.back());
    }
        norm_ = std::make_shared<std::remove_reference_t<decltype(*norm_)>>(runtime, hidden_size, rms_norm_eps, dtype);
        this->bind_child("norm", norm_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

void Ernie45Model::replace_embeddings(infinilm::runtime::Runtime &runtime,
                                      infinilm::runtime::Tensor inputs_embeds,
                                      const infinilm::runtime::Tensor &vision_hidden,
                                      const infinilm::runtime::Tensor &image_bound) const {
    auto bounds_cpu = image_bound->device().type() == infinilm::runtime::Device::Type::kCpu
        ? image_bound : image_bound->to(runtime, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    auto bounds = reinterpret_cast<const int64_t *>(bounds_cpu->data());
    const int64_t start = bounds[0];
    const int64_t end = bounds[1];
    if (start < 0 || end < start) {
        throw std::runtime_error("Ernie45Model: invalid image_bound");
    }
    const size_t span = static_cast<size_t>(end - start);
    if (vision_hidden->size(0) != span) {
        throw std::runtime_error("Ernie45Model: image feature length does not match image token span");
    }

    ASSERT_EQ(inputs_embeds->size(0), 1);
    auto out_slice = inputs_embeds->squeeze(0);
    out_slice->narrow({{0, static_cast<size_t>(start), span}})->copy_from(runtime, vision_hidden);
}

void Ernie45Model::apply_image_embeddings(infinilm::runtime::Tensor inputs_embeds,
                                          const InfinilmModel::Input &input,
                                          const Ernie45VisionModel &vision_model,
                                          global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    if (!input.image_bound.has_value() || !input.tgt_sizes.has_value()) {
        throw std::runtime_error("Ernie45Model: image_bound and tgt_sizes/grid_thw must be provided with pixel_values");
    }
    if (!resampler_model_) {
        throw std::runtime_error("Ernie45Model: resampler_model is required for image input");
    }
    const auto &pixel_values = input.pixel_values.value();
    const auto &image_bound = input.image_bound.value();
    const auto &grid_thw = input.tgt_sizes.value();
    if (pixel_values.size() != image_bound.size() || pixel_values.size() != grid_thw.size()) {
        throw std::runtime_error("Ernie45Model: pixel_values, image_bound and grid_thw must have the same number of elements");
    }
    auto &mm_metadata = forward_context.mm_metadata;
    if (!mm_metadata.image_req_ids.has_value()) {
        throw std::runtime_error("Ernie45Model: image_req_ids must be provided with pixel_values");
    }
    const auto &image_req_ids = mm_metadata.image_req_ids.value();
    if (image_req_ids.size() != pixel_values.size()) {
        throw std::runtime_error("Ernie45Model: image_req_ids count must match pixel_values count");
    }
    if (!input.input_offsets.has_value()) {
        throw std::runtime_error("Ernie45Model: input_offsets is required for multimodal replacement");
    }

    auto input_offsets_cpu = input.input_offsets.value()->device().type() == infinilm::runtime::Device::Type::kCpu
        ? input.input_offsets.value()
        : input.input_offsets.value()->to(runtime, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    auto *offsets = reinterpret_cast<const int32_t *>(input_offsets_cpu->data());
    std::vector<size_t> visual_token_ranges;
    for (size_t image_idx = 0; image_idx < pixel_values.size(); ++image_idx) {
        const size_t req_id = image_req_ids[image_idx];
        auto bounds_cpu = image_bound[image_idx]->device().type() == infinilm::runtime::Device::Type::kCpu
            ? image_bound[image_idx]
            : image_bound[image_idx]->to(runtime, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
        auto bounds = reinterpret_cast<const int64_t *>(bounds_cpu->data());
        const size_t packed_start = static_cast<size_t>(offsets[req_id]) + static_cast<size_t>(bounds[0]);
        const size_t packed_end = static_cast<size_t>(offsets[req_id]) + static_cast<size_t>(bounds[1]);
        visual_token_ranges.push_back(packed_start);
        visual_token_ranges.push_back(packed_end);
        auto req_embeds = inputs_embeds->narrow({{1, static_cast<size_t>(offsets[req_id]), static_cast<size_t>(offsets[req_id + 1] - offsets[req_id])}});
        auto image_features = vision_model.forward(runtime, pixel_values[image_idx], grid_thw[image_idx]);
        auto vision_hidden = resampler_model_->forward(runtime, image_features, grid_thw[image_idx]);
        replace_embeddings(runtime, req_embeds, vision_hidden, image_bound[image_idx]);
    }
    mm_metadata.visual_token_ranges = std::move(visual_token_ranges);
}

infinilm::runtime::Tensor Ernie45Model::forward_embeds(infinilm::runtime::Tensor hidden_states,
                                                const infinilm::runtime::Tensor &positions,
                                                global_state::ForwardContext &forward_context) const {
    infinilm::runtime::Tensor residual;
    for (auto &layer : layers_) {
        layer->forward(positions, hidden_states, residual, forward_context);
    }
    norm_->forward_inplace(forward_context.require_runtime(), hidden_states, residual);
    return hidden_states;
}

infinilm::runtime::Tensor Ernie45Model::forward(const InfinilmModel::Input &input,
                                         global_state::ForwardContext &forward_context,
                                         const Ernie45VisionModel *vision_model) const {
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto hidden_states = embed_tokens_->forward(forward_context.require_runtime(), input_ids);
    if (input.pixel_values.has_value() && !input.pixel_values.value().empty()) {
        if (vision_model == nullptr) {
            throw std::runtime_error("Ernie45Model: vision_model is required for image input");
        }
        apply_image_embeddings(hidden_states, input, *vision_model, forward_context);
    }
    return forward_embeds(hidden_states, positions, forward_context);
}

} // namespace infinilm::models::ernie4_5_vl
