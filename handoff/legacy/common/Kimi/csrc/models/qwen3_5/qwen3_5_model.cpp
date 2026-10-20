#include "qwen3_5_model.hpp"
#include "../models_registry.hpp"
#include <exception>

#include "../qwen3_next/qwen3_next_allocate_kv_cache_tensors.hpp"

#include <cstdint>
#include <cstring>
#include "runtime/runtime.hpp"
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::models::qwen3_5 {
namespace {

std::vector<int32_t> tensor_to_i32_vector(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &tensor) {
    if (!tensor || !tensor->is_contiguous()) {
        throw std::invalid_argument("Qwen35Model: metadata must be contiguous");
    }
    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    auto read = [&](void *bytes) {
        if (tensor->device() == cpu) {
            std::memcpy(bytes, tensor->data(), tensor->nbytes());
        } else if (tensor->device() == runtime.device()) {
            runtime.memcpyD2H(bytes, tensor->data(), tensor->nbytes());
            runtime.syncStream();
        } else {
            throw std::invalid_argument("Qwen35Model: metadata belongs to another device");
        }
    };
    std::vector<int32_t> values(tensor->numel());
    if (tensor->dtype() == infinilm::runtime::DataType::kInt32) {
        read(values.data());
    } else if (tensor->dtype() == infinilm::runtime::DataType::kInt64) {
        std::vector<int64_t> wide(tensor->numel());
        read(wide.data());
        for (size_t i = 0; i < wide.size(); ++i) {
            values[i] = static_cast<int32_t>(wide[i]);
        }
    } else {
        throw std::runtime_error("Qwen35Model: expected int32 or int64 tensor");
        }
        return values;
}

} // namespace

Qwen35ModelBase::Qwen35ModelBase(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config)
    : model_config_(model_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info, attention_backend, ep_config);
    try {
    const auto &dtype{model_config->get_dtype()};
    nlohmann::json &config_json = model_config->get_config_json();

    if (config_json.contains("vision_config") && !config_json["vision_config"].is_null()) {
            visual_ = this->bind_child("visual", std::make_shared<std::remove_reference_t<decltype(*visual_ )>>(runtime, config_json["vision_config"], dtype));
        }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

void Qwen35ModelBase::replace_image_embeddings(infinilm::runtime::Tensor &inputs_embeds,
                                               const InfinilmModel::Input &input,
                                               global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::invalid_argument("Qwen35Model: dynamic visual metadata is not recording-qualified");
    }
    if (!input.pixel_values.has_value() || input.pixel_values->empty()) {
        return;
    }
    if (!input.image_grid_thw.has_value()) {
        throw std::runtime_error("Qwen35Model: image_grid_thw must be provided with pixel_values");
    }
    if (!input.image_bound.has_value()) {
        throw std::runtime_error("Qwen35Model: image_bound must be provided with pixel_values");
    }
    if (!input.input_offsets.has_value()) {
        throw std::runtime_error("Qwen35Model: input_offsets are required for image replacement");
    }
    if (!visual_) {
        throw std::runtime_error("Qwen35Model: visual module is not initialized");
    }

    const auto &pixel_values = input.pixel_values.value();
    const auto &image_grid_thw = input.image_grid_thw.value();
    const auto &image_bound = input.image_bound.value();
    if (pixel_values.size() != image_grid_thw.size() || pixel_values.size() != image_bound.size()) {
        throw std::runtime_error("Qwen35Model: pixel_values, image_grid_thw and image_bound size mismatch");
    }

    const auto &image_req_ids_opt = forward_context.mm_metadata.image_req_ids;
    if (!image_req_ids_opt.has_value() || image_req_ids_opt->size() != pixel_values.size()) {
        throw std::runtime_error("Qwen35Model: image_req_ids must match pixel_values");
    }

    auto offsets = tensor_to_i32_vector(runtime, input.input_offsets.value());

    for (size_t image_idx = 0; image_idx < pixel_values.size(); ++image_idx) {
        const size_t req_id = image_req_ids_opt.value().at(image_idx);
        if (req_id + 1 >= offsets.size()) {
            throw std::runtime_error("Qwen35Model: image request id is out of range");
        }

        auto bound = tensor_to_i32_vector(runtime, image_bound.at(image_idx));
        if (bound.size() < 2) {
            throw std::runtime_error("Qwen35Model: image_bound must contain start and end positions");
        }
        if (bound[0] < 0 || bound[1] < bound[0]) {
            throw std::runtime_error("Qwen35Model: invalid image_bound range");
        }

        auto vision_hidden = visual_->forward(runtime, pixel_values.at(image_idx), image_grid_thw.at(image_idx));
        const size_t vision_len = vision_hidden->size(0);
        const size_t req_start = static_cast<size_t>(offsets[req_id]);
        const size_t req_len = static_cast<size_t>(offsets[req_id + 1] - offsets[req_id]);
        const size_t local_start = static_cast<size_t>(bound[0]);
        const size_t local_end = static_cast<size_t>(bound[1]);
        const size_t span_len = local_end - local_start;
        if (local_end > req_len) {
            throw std::runtime_error("Qwen35Model: image_bound is out of request range");
        }
        if (span_len != vision_len) {
            throw std::runtime_error("Qwen35Model: image_bound length does not match visual embedding length");
        }

        inputs_embeds->narrow({{1, req_start + local_start, span_len}})
            ->copy_from(runtime, vision_hidden->unsqueeze(0));
    }
}

void Qwen35ModelBase::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank_info) {
    cache::validate_backend_cache_config(cache_config, attention_backend);
    if (cache_config != nullptr) {
        cache::validate_cache_runtime(forward_context.require_runtime(), rank_info);
    }
    forward_context.kv_cache_vec.clear();
    forward_context.conv_state_vec.clear();
    forward_context.ssm_state_vec.clear();
    if (nullptr == cache_config) {
        return;
    }

    auto cache_vectors = infinilm::models::qwen3_next::qwen3_next_allocate_cache_tensors(
        forward_context.require_runtime(), rank_info, cache_config, model_config_, attention_backend);
    forward_context.kv_cache_vec = std::move(cache_vectors.kv_cache_tensors);
    forward_context.conv_state_vec = std::move(cache_vectors.conv_state_tensors);
    forward_context.ssm_state_vec = std::move(cache_vectors.ssm_state_tensors);
}

} // namespace infinilm::models::qwen3_5
