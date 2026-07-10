#include "ernie4_5_model.hpp"

#include "../../global_state/global_state.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::ernie4_5_vl {

Ernie45Model::Ernie45Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device)
    : model_config_(model_config) {
    const auto &dtype = model_config->get_dtype();
    auto &config_json = model_config->get_config_json();
    if (config_json.contains("vision_config") && config_json["vision_config"].is_object()) {
        INFINICORE_NN_MODULE_INIT(resampler_model, config_json, dtype, device);
    }
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt, dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<Ernie45DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, rms_norm_eps, dtype, device);
}

void Ernie45Model::replace_embeddings(infinicore::Tensor inputs_embeds,
                                      const infinicore::Tensor &vision_hidden,
                                      const infinicore::Tensor &image_bound) const {
    auto bounds_cpu = image_bound->to(infinicore::Device::cpu());
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
    out_slice->narrow({{0, static_cast<size_t>(start), span}})->copy_from(vision_hidden);
}

void Ernie45Model::apply_image_embeddings(infinicore::Tensor inputs_embeds,
                                          const InfinilmModel::Input &input,
                                          const Ernie45VisionModel &vision_model) const {
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
    auto &mm_metadata = global_state::get_forward_context().mm_metadata;
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

    auto input_offsets_cpu = input.input_offsets.value()->to(infinicore::Device::cpu());
    auto *offsets = reinterpret_cast<const int32_t *>(input_offsets_cpu->data());
    for (size_t image_idx = 0; image_idx < pixel_values.size(); ++image_idx) {
        const size_t req_id = image_req_ids[image_idx];
        auto req_embeds = inputs_embeds->narrow({{1, static_cast<size_t>(offsets[req_id]), static_cast<size_t>(offsets[req_id + 1] - offsets[req_id])}});
        auto image_features = vision_model.forward(pixel_values[image_idx], grid_thw[image_idx]);
        auto vision_hidden = resampler_model_->forward(image_features, grid_thw[image_idx]);
        replace_embeddings(req_embeds, vision_hidden, image_bound[image_idx]);
    }
}

infinicore::Tensor Ernie45Model::forward_embeds(infinicore::Tensor hidden_states,
                                                const infinicore::Tensor &positions) const {
    infinicore::Tensor residual;
    for (auto &layer : layers_) {
        layer->forward(positions, hidden_states, residual);
    }
    norm_->forward_inplace(hidden_states, residual);
    return hidden_states;
}

infinicore::Tensor Ernie45Model::forward(const InfinilmModel::Input &input) const {
    return forward(input, nullptr);
}

infinicore::Tensor Ernie45Model::forward(const InfinilmModel::Input &input,
                                         const Ernie45VisionModel *vision_model) const {
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto hidden_states = embed_tokens_->forward(input_ids);
    if (input.pixel_values.has_value() && !input.pixel_values.value().empty()) {
        if (vision_model == nullptr) {
            throw std::runtime_error("Ernie45Model: vision_model is required for image input");
        }
        apply_image_embeddings(hidden_states, input, *vision_model);
    }
    return forward_embeds(hidden_states, positions);
}

void Ernie45Model::reset_cache(const cache::CacheConfig *) {
    // Cache tensors are allocated by Ernie45ForConditionalGeneration, which inherits InfinilmModel.
}

} // namespace infinilm::models::ernie4_5_vl
