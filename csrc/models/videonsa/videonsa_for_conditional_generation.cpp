#include "videonsa_for_conditional_generation.hpp"
#include "../../weights/weights.hpp"
#include "../../execution/tensor_expressions.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include <exception>
#include "runtime/runtime.hpp"
#include <cstddef>
#include <vector>
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
    if (text_config_json["num_attention_heads"].get<int64_t>() <= 0)
        throw std::invalid_argument("VideoNSA requires positive attention heads");
    if (!text_config_json.contains("head_dim")) {
        text_config_json["head_dim"] = text_config_json["hidden_size"].get<size_t>() / text_config_json["num_attention_heads"].get<size_t>();
    }
    if (!text_config_json.contains("attention_bias")) {
        text_config_json["attention_bias"] = true;
    }
    return std::make_shared<infinilm::config::ModelConfig>(text_config_json);
}

// Metadata downloads use the execution owner, never the caller's TLS stream.
infinilm::runtime::Tensor host_metadata(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &input) {
    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    if (input->device() == cpu) return input;
    if (input->device() != runtime.device())
        throw std::invalid_argument("VideoNSA metadata belongs to another device");
    auto dense = input->contiguous(runtime);
    auto storage = std::make_shared<std::vector<std::byte>>(dense->nbytes());
    runtime.memcpyD2H(storage->data(), dense->data(), storage->size());
    return infinilm::runtime::Tensor::from_blob(storage->data(), dense->shape(), dense->dtype(), cpu, storage);
}

} // namespace

VideoNSAForConditionalGeneration::VideoNSAForConditionalGeneration(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank,
    backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep) {
    models::validate_runtime_construction(model_config, runtime, rank);
    try {
    model_config_ = model_config;
    auto text_config = text_config_from(model_config);
    const size_t hidden_size = text_config->get<size_t>("hidden_size");
    const size_t vocab_size = text_config->get<size_t>("vocab_size");
    const auto &dtype{text_config->get_dtype()};

    model_ = this->bind_child("model", std::make_shared<std::remove_reference_t<decltype(*model_ )>>(runtime, text_config, rank, attention_backend, ep));
    visual_ = this->bind_child("visual", std::make_shared<std::remove_reference_t<decltype(*visual_ )>>(runtime, model_config->get_config_json()["vision_config"], dtype));
    lm_head_ = this->bind_child("lm_head", std::make_shared<std::remove_reference_t<decltype(*lm_head_ )>>(runtime, hidden_size, vocab_size, false, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

void VideoNSAForConditionalGeneration::replace_embeddings(infinilm::runtime::Runtime &runtime,
                                                          infinilm::runtime::Tensor inputs_embeds,
                                                          const infinilm::runtime::Tensor &vision_hidden,
                                                          const infinilm::runtime::Tensor &image_bound) const {
    auto bounds_cpu = host_metadata(runtime, image_bound);
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
        out_slice->narrow({{0, size_t(start), len}})->copy_from(runtime, patch_embed);
        vision_offset += len;
    }
}

infinilm::InfinilmModel::Output VideoNSAForConditionalGeneration::forward(
    const infinilm::InfinilmModel::Input &input, global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    if (input.pixel_values.has_value() && input.pixel_values.value().size() > 0) {
        if (!input.image_bound.has_value() || !input.tgt_sizes.has_value()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: image_bound and tgt_sizes must be provided with pixel_values");
        }
        auto input_ids = input.input_ids.value();
        auto inputs_embeds = model_->embed_tokens(input_ids, forward_context);
        auto input_offsets_cpu = host_metadata(runtime, input.input_offsets.value());
        int32_t *offsets = reinterpret_cast<int32_t *>(input_offsets_cpu->data());

        const auto &image_req_ids = forward_context.mm_metadata.image_req_ids.value();
        if (input.pixel_values->size() != image_req_ids.size() || input.image_bound->size() != image_req_ids.size() || input.tgt_sizes->size() != image_req_ids.size()) {
            throw std::runtime_error("VideoNSAForConditionalGeneration: multimodal tensor lists must match image_req_ids");
        }

        std::vector<infinilm::runtime::Tensor> pixel_tensors;
        std::vector<infinilm::runtime::Tensor> grid_tensors;
        pixel_tensors.reserve(image_req_ids.size());
        grid_tensors.reserve(image_req_ids.size());
        for (size_t media_idx = 0; media_idx < image_req_ids.size(); ++media_idx) {
            pixel_tensors.push_back(input.pixel_values.value().at(media_idx));
            grid_tensors.push_back(input.tgt_sizes.value().at(media_idx));
        }
        infinilm::runtime::Tensor batched_pixels;
        if (pixel_tensors.size() == 1) {
            batched_pixels = pixel_tensors.front();
        } else {
            batched_pixels = infinilm::execution::cat_alloc(runtime, pixel_tensors, 0);
        }
        infinilm::runtime::Tensor batched_grids;
        if (grid_tensors.size() == 1) {
            batched_grids = grid_tensors.front();
        } else {
            batched_grids = infinilm::execution::cat_alloc(runtime, grid_tensors, 0);
        }
        auto batched_vision_hidden = visual_->forward(runtime, batched_pixels, batched_grids);

        size_t vision_offset = 0;
        for (size_t media_idx = 0; media_idx < image_req_ids.size(); ++media_idx) {
            const size_t req_id = image_req_ids[media_idx];
            auto bounds_cpu = host_metadata(runtime, input.image_bound.value().at(media_idx))->squeeze(0);
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
            replace_embeddings(runtime, req_embeds, vision_hidden, input.image_bound.value().at(media_idx));
            vision_offset += vision_len;
        }

        auto hidden_states = model_->forward_embeds(inputs_embeds, input.position_ids.value(), forward_context);
        auto logits = lm_head_->forward(runtime, hidden_states);
        return {logits};
    }

    auto hidden_states = model_->forward(input, forward_context);
    auto logits = lm_head_->forward(runtime, hidden_states);
    return {logits};
}

void VideoNSAForConditionalGeneration::prepare_paged_owner(global_state::ForwardContext &context) const {
    // Traverse actual Module ownership, only this model's NSA attention type.
    const auto visit = [&](const auto &self, const infinilm::weights::ModelWeights *module) -> void {
        if (const auto *attention = dynamic_cast<const VideoNSAAttention *>(module))
            attention->prepare_paged_owner(context);
        for (const auto &[name, child] : module->children()) {
            (void)name;
            self(self, child.get());
        }
    };
    visit(visit, this);
}
void VideoNSAForConditionalGeneration::finish_paged_owner(bool completed) const {
    // Traverse actual Module ownership, only this model's NSA attention type.
    const auto visit = [&](const auto &self, const infinilm::weights::ModelWeights *module) -> void {
        if (const auto *attention = dynamic_cast<const VideoNSAAttention *>(module))
            attention->finish_paged_owner(completed);
        for (const auto &[name, child] : module->children()) {
            (void)name;
            self(self, child.get());
        }
    };
    visit(visit, this);
}
void VideoNSAForConditionalGeneration::invalidate_paged_owner() const noexcept {
    // Traverse actual Module ownership, only this model's NSA attention type.
    const auto visit = [&](const auto &self, const infinilm::weights::ModelWeights *module) -> void {
        if (const auto *attention = dynamic_cast<const VideoNSAAttention *>(module))
            attention->invalidate_paged_owner();
        for (const auto &[name, child] : module->children()) {
            (void)name;
            self(self, child.get());
        }
    };
    visit(visit, this);
}

void VideoNSAForConditionalGeneration::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank) {
    cache::validate_backend_cache_config(cache_config, attention_backend);
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr, forward_context, attention_backend, rank);
        return;
    }
    auto &runtime = forward_context.require_runtime();
    cache::validate_cache_runtime(runtime, rank);
    cache_config_ = cache_config->unique_copy();

    auto text_config = text_config_from(model_config_);
    auto &kv_cache_vec = forward_context.kv_cache_vec;
    kv_cache_vec.clear();
    kv_cache_vec = std::move(default_allocate_kv_cache_tensors(runtime, rank, cache_config, text_config, attention_backend));
}

std::shared_ptr<infinilm::config::ModelConfig> create_videonsa_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("videonsa" != model_type) {
        throw std::runtime_error("infinilm::models::videonsa::create_videonsa_model_config: model_type is not videonsa");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (config_json.contains("text_config")) {
        nlohmann::json &text_config_json = config_json["text_config"];
        if (text_config_json["num_attention_heads"].get<int64_t>() <= 0)
            throw std::invalid_argument("VideoNSA requires positive attention heads");
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
        if (model_config->get<int64_t>("num_attention_heads") <= 0)
            throw std::invalid_argument("VideoNSA requires positive attention heads");
        if (!config_json.contains("head_dim")) {
            config_json["head_dim"] = model_config->get<size_t>("hidden_size") / model_config->get<size_t>("num_attention_heads");
        }
        if (!config_json.contains("attention_bias")) {
            config_json["attention_bias"] = true;
        }
    }
    if (config_json.contains("vision_config")) {
        const auto &vision = config_json["vision_config"];
        if (vision.value("depth", 32) > 0) {
            const int width = vision.value("hidden_size", 1280), heads = vision.value("num_heads", 16);
            if (width <= 0 || heads <= 0 || width % heads != 0)
                throw std::invalid_argument("VideoNSA vision head geometry is invalid");
        }
    }
    return model_config;
}

} // namespace infinilm::models::videonsa

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    videonsa,
    infinilm::models::videonsa::VideoNSAForConditionalGeneration,
    infinilm::models::videonsa::create_videonsa_model_config);
} // namespace
