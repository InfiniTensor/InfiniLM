#include "minicpmv_model.hpp"
#include "minicpmv_media_mapping.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "runtime/runtime.hpp"
#include <exception>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infinilm::models::minicpmv {

namespace {
template <typename T>
std::vector<T> copy_cpu_metadata(const infinilm::runtime::Tensor &tensor,
                                 infinilm::runtime::DataType dtype,
                                 infinilm::runtime::Runtime &runtime,
                                 const char *name) {
    if (!tensor || tensor->dtype() != dtype || !tensor->is_contiguous()) {
        throw std::invalid_argument(std::string("MiniCPMVModel: ") + name
                                    + " has invalid dtype or layout");
    }
    std::vector<T> values(tensor->numel());
    if (values.empty()) {
        return values;
    }
    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    if (tensor->device() == cpu) {
        std::memcpy(values.data(), tensor->data(), tensor->nbytes());
    } else if (tensor->device() == runtime.device()) {
        runtime.memcpyD2H(values.data(), tensor->data(), tensor->nbytes());
    } else {
        throw std::invalid_argument(std::string("MiniCPMVModel: ") + name
                                    + " is on a different device from the explicit Runtime");
    }
    return values;
}
} // namespace

MiniCPMVModel::MiniCPMVModel(infinilm::runtime::Runtime &runtime,
                             std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const engine::distributed::RankInfo &rank,
                             backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep)
    : config_(model_config) {
    models::validate_runtime_construction(model_config, runtime, rank);
    try {

        const auto &dtype{model_config->get_dtype()};

        llm_ = this->bind_child("llm", std::make_shared<std::remove_reference_t<decltype(*llm_)>>(runtime, model_config, rank, attention_backend, ep));

        // Use get_ref instead of get to get a reference
        auto &vision_cfg = model_config->get_ref("vision_config");

        if (model_config->get<bool>("drop_vision_last_layer") && vision_cfg.value("num_hidden_layers", 0) > 0) {
            vision_cfg["num_hidden_layers"] = vision_cfg.value("num_hidden_layers", 0) - 1;
        }

        vpm_ = this->bind_child("vpm", std::make_shared<std::remove_reference_t<decltype(*vpm_)>>(runtime, vision_cfg, dtype, false));

        size_t embed_dim = model_config->get<size_t>("hidden_size");
        size_t num_heads = embed_dim / 128;
        resampler_ = this->bind_child("resampler", std::make_shared<std::remove_reference_t<decltype(*resampler_)>>(runtime, model_config->get<size_t>("query_num"), embed_dim, num_heads, vision_cfg.value("hidden_size", 768), vision_cfg.value("image_size", 224), vision_cfg.value("patch_size", 16), dtype));

    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

void MiniCPMVModel::replace_embeddings(infinilm::runtime::Tensor inputs_embeds,
                                       const infinilm::runtime::Tensor &vision_hidden,
                                       const std::vector<int64_t> &image_bounds,
                                       infinilm::runtime::Runtime &runtime) const {
    auto batch_size = inputs_embeds->size(0);

    ASSERT_EQ(batch_size, 1);
    auto out_slice = inputs_embeds->squeeze(0);
    auto vision_len = vision_hidden->size(0);
    for (size_t patch = 0; patch < vision_len; ++patch) {
        auto patch_embed = vision_hidden->narrow({{0, patch, 1}})->squeeze(0);
        auto start = image_bounds[2 * patch];
        auto end = image_bounds[2 * patch + 1];

        out_slice->narrow({{0, size_t(start), size_t(end - start)}})->copy_from(runtime, patch_embed);
    }
}

InfinilmModel::Output MiniCPMVModel::forward(const InfinilmModel::Input &input,
                                             global_state::ForwardContext &forward_context) const {
    if (forward_context.runtime == nullptr) {
        throw std::runtime_error("MiniCPMVModel: runtime is not initialized");
    }
    forward_context.runtime->assertOwnerThread();
    if (!attention_backend_.has_value()) {
        throw std::runtime_error("MiniCPMVModel: reset_cache must prepare the attention backend before forward");
    }
    if (!input.input_ids.has_value()) {
        throw std::runtime_error("MiniCPMVModel: input_ids is required");
    }
    auto input_ids = input.input_ids.value();
    if (!input_ids || input_ids->device() != forward_context.runtime->device()) {
        throw std::invalid_argument("MiniCPMVModel: input_ids must be on the explicit Runtime device");
    }

    if (input.pixel_values.has_value() && input.pixel_values.value().size() > 0) {
        if (forward_context.runtime->isGraphRecording()) {
            throw std::invalid_argument("MiniCPMVModel: dynamic visual metadata is not recording-qualified");
        }
        if (!input.image_bound.has_value() or !input.tgt_sizes.has_value()) {
            throw std::runtime_error("MiniCPMVModel: image_bound and tgt_sizes must be provided with pixel_values");
        }
        if (input.pixel_values->size() != input.image_bound->size() || input.pixel_values->size() != input.tgt_sizes->size()) {
            throw std::runtime_error("MiniCPMVModel: pixel_values, image_bound and tgt_sizes must have the same number of elements");
        }
        if (!input.input_offsets.has_value()) {
            throw std::runtime_error("MiniCPMVModel: input_offsets is required with pixel_values");
        }
        const auto &image_req_ids = forward_context.mm_metadata.image_req_ids;
        if (!image_req_ids.has_value() || image_req_ids->size() != input.pixel_values->size()) {
            throw std::runtime_error("MiniCPMVModel: image_req_ids must match pixel_values");
        }

        if (input_ids->ndim() != 2 || input_ids->size(0) != 1) {
            throw std::invalid_argument("MiniCPMVModel: packed input_ids must have shape [1, tokens]");
        }
        const auto &offset_tensor = *input.input_offsets;
        if (!offset_tensor || offset_tensor->ndim() != 1) {
            throw std::invalid_argument("MiniCPMVModel: input_offsets must be a vector");
        }
        if (!input.past_sequence_lengths.has_value()
            || !*input.past_sequence_lengths
            || (*input.past_sequence_lengths)->ndim() != 1) {
            throw std::invalid_argument("MiniCPMVModel: cached lengths must be a vector");
        }
        auto offsets = copy_cpu_metadata<int32_t>(
            offset_tensor, infinilm::runtime::DataType::kInt32,
            *forward_context.runtime, "input_offsets");
        auto cached_lengths = copy_cpu_metadata<int32_t>(
            *input.past_sequence_lengths, infinilm::runtime::DataType::kInt32,
            *forward_context.runtime, "past_sequence_lengths");
        auto media_slices = map_media_to_requests(
            offsets, input_ids->size(1), *image_req_ids, cached_lengths);
        std::vector<std::vector<int64_t>> media_bounds;
        media_bounds.reserve(media_slices.size());
        for (const auto &slice : media_slices) {
            const auto &pixels = input.pixel_values->at(slice.media_index);
            const auto &target = input.tgt_sizes->at(slice.media_index);
            const auto &bounds = input.image_bound->at(slice.media_index);
            if (!pixels || pixels->ndim() != 4 || pixels->size(0) == 0
                || pixels->device() != forward_context.runtime->device()
                || !target || target->ndim() != 2 || target->size(1) != 2
                || target->size(0) != pixels->size(0)
                || target->dtype() != infinilm::runtime::DataType::kInt64
                || target->device() != forward_context.runtime->device()
                || !bounds || bounds->ndim() != 3 || bounds->size(0) != 1
                || bounds->size(2) != 2) {
                throw std::invalid_argument("MiniCPMVModel: invalid media tensor shape, dtype or device");
            }
            auto values = copy_cpu_metadata<int64_t>(
                bounds, infinilm::runtime::DataType::kInt64,
                *forward_context.runtime, "image_bound");
            validate_media_bounds(values, pixels->size(0), slice.token_count);
            media_bounds.push_back(std::move(values));
        }

        auto inputs_embeds = llm_->model().embed_tokens(input_ids, forward_context);

        // inputs_embeds concat tokens from all requests, while images are processed per request
        // slice inputs_embeds using request offsets to get the embedding of each request
        for (const auto &slice : media_slices) {
            auto pixel_values = input.pixel_values->at(slice.media_index);
            auto target_sizes = input.tgt_sizes->at(slice.media_index);
            auto vision_embedding = vpm_->forward(*forward_context.runtime, pixel_values, target_sizes, *attention_backend_);
            auto vision_hidden = resampler_->forward(*forward_context.runtime, vision_embedding, target_sizes);
            if (!vision_hidden || vision_hidden->size(0) != pixel_values->size(0)) {
                throw std::runtime_error("MiniCPMVModel: vision patch count differs from media metadata");
            }
            replace_embeddings(inputs_embeds->narrow({{1, slice.token_offset, slice.token_count}}),
                               vision_hidden, media_bounds[slice.media_index], *forward_context.runtime);
        }

        auto hidden_states = llm_->model().forward_embeds(
            inputs_embeds,
            input.position_ids.value(),
            forward_context);

        auto logits = llm_->logits_from_hidden(forward_context.require_runtime(), hidden_states);
        return {logits};
    }

    return llm_->forward(input, forward_context);
}

void MiniCPMVModel::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank) {
    if (cache_config != nullptr) {
        cache::validate_cache_runtime(forward_context.require_runtime(), rank);
    }
    attention_backend_.reset();
    llm_->reset_cache(cache_config, forward_context, attention_backend, rank);
    attention_backend_ = attention_backend;
}

std::shared_ptr<infinilm::config::ModelConfig> create_minicpmv_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("minicpmv" != model_type) {
        throw std::runtime_error("infinilm::models::minicpmv::create_minicpmv_model_config: model_type is not minicpmv");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    if (model_config->get<int64_t>("num_attention_heads") <= 0) {
        throw std::invalid_argument("MiniCPMV requires positive attention heads");
    }
    if (!config_json.contains("head_dim")) {
        size_t head_dim = model_config->get<size_t>("hidden_size")
                        / model_config->get<size_t>("num_attention_heads");
        config_json["head_dim"] = head_dim;
    }

    models::validate_dense_construction_metadata(model_config, "MiniCPMV text");
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t heads = hidden / 128;
    if (heads == 0 || hidden % heads != 0) {
        throw std::invalid_argument("MiniCPMV resampler head geometry is invalid");
    }
    const auto &vision = model_config->get_ref("vision_config");
    if (vision.value("patch_size", 16) <= 0) {
        throw std::invalid_argument("MiniCPMV requires positive vision patch_size");
    }
    const int layers = vision.value("num_hidden_layers", 12);
    const bool drop = model_config->get<bool>("drop_vision_last_layer");
    if (layers > (drop ? 1 : 0)) {
        const int width = vision.value("hidden_size", 768), count = vision.value("num_attention_heads", 12);
        if (width <= 0 || count <= 0 || width % count != 0) {
            throw std::invalid_argument("MiniCPMV vision head geometry is invalid");
        }
    }
    // Do not apply drop_vision_last_layer here: the constructor retains its one mutation.
    return model_config;
}

} // namespace infinilm::models::minicpmv

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    minicpmv,
    infinilm::models::minicpmv::MiniCPMVModel,
    infinilm::models::minicpmv::create_minicpmv_model_config);
} // namespace
