#include "qwen3_5_model.hpp"

#include "../../global_state/global_state.hpp"
#include "../qwen3_next/qwen3_next_allocate_kv_cache_tensors.hpp"

#include <infinicore/ops/cat.hpp>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::models::qwen3_5 {
namespace {

std::vector<int32_t> tensor_to_i32_vector(const infinicore::Tensor &tensor) {
    auto cpu_tensor = tensor->to(infinicore::Device::cpu());
    std::vector<int32_t> values(cpu_tensor->numel());
    if (cpu_tensor->dtype() == infinicore::DataType::I32) {
        const auto *ptr = reinterpret_cast<const int32_t *>(cpu_tensor->data());
        values.assign(ptr, ptr + cpu_tensor->numel());
        return values;
    }
    if (cpu_tensor->dtype() == infinicore::DataType::I64) {
        const auto *ptr = reinterpret_cast<const int64_t *>(cpu_tensor->data());
        for (size_t i = 0; i < cpu_tensor->numel(); ++i) {
            values[i] = static_cast<int32_t>(ptr[i]);
        }
        return values;
    }
    throw std::runtime_error("Qwen35Model: expected int32 or int64 tensor");
}

} // namespace

Qwen35ModelBase::Qwen35ModelBase(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device)
    : model_config_(model_config) {
    const auto &dtype{model_config->get_dtype()};
    nlohmann::json &config_json = model_config->get_config_json();

    if (config_json.contains("vision_config") && !config_json["vision_config"].is_null()) {
        INFINICORE_NN_MODULE_INIT(visual, config_json["vision_config"], dtype, device);
    }
}

void Qwen35ModelBase::replace_image_embeddings(infinicore::Tensor &inputs_embeds,
                                               const InfinilmModel::Input &input) const {
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

    const auto &image_req_ids_opt = infinilm::global_state::get_forward_context().mm_metadata.image_req_ids;
    if (!image_req_ids_opt.has_value() || image_req_ids_opt->size() != pixel_values.size()) {
        throw std::runtime_error("Qwen35Model: image_req_ids must match pixel_values");
    }

    auto offsets = tensor_to_i32_vector(input.input_offsets.value());
    const auto &vision_config = model_config_->get_config_json().at("vision_config");
    const size_t spatial_merge_size = vision_config.at("spatial_merge_size").get<size_t>();
    std::vector<std::vector<int32_t>> grids(pixel_values.size());
    std::vector<std::vector<int32_t>> bounds(pixel_values.size());
    bool can_batch_images = pixel_values.size() > 1;
    const auto first_pixel_shape = pixel_values.front()->shape();
    const auto first_grid_shape = image_grid_thw.front()->shape();

    for (size_t image_idx = 0; image_idx < pixel_values.size(); ++image_idx) {
        const size_t req_id = infinilm::global_state::get_forward_context()
                                  .mm_metadata.image_req_ids.value()
                                  .at(image_idx);
        if (req_id + 1 >= offsets.size()) {
            throw std::runtime_error("Qwen35Model: image request id is out of range");
        }
        bounds[image_idx] = tensor_to_i32_vector(image_bound.at(image_idx));
        if (bounds[image_idx].size() < 2 || bounds[image_idx][0] < 0
            || bounds[image_idx][1] < bounds[image_idx][0]) {
            throw std::runtime_error("Qwen35Model: invalid image_bound range");
        }
        grids[image_idx] = tensor_to_i32_vector(image_grid_thw.at(image_idx));
        if (pixel_values.at(image_idx)->shape() != first_pixel_shape
            || image_grid_thw.at(image_idx)->shape() != first_grid_shape
            || grids[image_idx] != grids.front()) {
            can_batch_images = false;
        }
    }

    if (can_batch_images) {
        std::vector<infinicore::Tensor> batched_grids;
        batched_grids.reserve(image_grid_thw.size());
        for (const auto &grid : image_grid_thw) {
            batched_grids.push_back(grid->unsqueeze(0));
        }
        auto vision_hidden = visual_->forward(
            infinicore::op::cat(pixel_values, 0),
            infinicore::op::cat(batched_grids, 0));
        size_t vision_offset = 0;
        for (size_t image_idx = 0; image_idx < pixel_values.size(); ++image_idx) {
            const size_t req_id = infinilm::global_state::get_forward_context()
                                      .mm_metadata.image_req_ids.value()
                                      .at(image_idx);
            const size_t req_start = static_cast<size_t>(offsets[req_id]);
            const size_t req_len = static_cast<size_t>(
                offsets[req_id + 1] - offsets[req_id]);
            const size_t local_start = static_cast<size_t>(bounds[image_idx][0]);
            const size_t local_end = static_cast<size_t>(bounds[image_idx][1]);
            const size_t span_len = local_end - local_start;
            if (local_end > req_len) {
                throw std::runtime_error("Qwen35Model: image_bound is out of request range");
            }
            const auto &grid = grids[image_idx];
            const size_t vision_len = static_cast<size_t>(grid[0])
                * (static_cast<size_t>(grid[1]) / spatial_merge_size)
                * (static_cast<size_t>(grid[2]) / spatial_merge_size);
            if (span_len != vision_len
                || vision_offset + vision_len > vision_hidden->size(0)) {
                throw std::runtime_error(
                    "Qwen35Model: image_bound length does not match visual embedding length");
            }
            inputs_embeds->narrow({{1, req_start + local_start, span_len}})
                ->copy_from(vision_hidden->narrow({{0, vision_offset, vision_len}})
                                ->unsqueeze(0));
            vision_offset += vision_len;
        }
        return;
    }

    for (size_t image_idx = 0; image_idx < pixel_values.size(); ++image_idx) {
        const size_t req_id = infinilm::global_state::get_forward_context()
                                  .mm_metadata.image_req_ids.value()
                                  .at(image_idx);
        if (req_id + 1 >= offsets.size()) {
            throw std::runtime_error("Qwen35Model: image request id is out of range");
        }
        auto bound = bounds.at(image_idx);
        if (bound.size() < 2) {
            throw std::runtime_error("Qwen35Model: image_bound must contain start and end positions");
        }
        if (bound[0] < 0 || bound[1] < bound[0]) {
            throw std::runtime_error("Qwen35Model: invalid image_bound range");
        }
        auto vision_hidden = visual_->forward(pixel_values.at(image_idx), image_grid_thw.at(image_idx));
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
            ->copy_from(vision_hidden->unsqueeze(0));
    }
}
void Qwen35ModelBase::reset_cache(const cache::CacheConfig *cache_config) {
    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.kv_cache_vec.clear();
    forward_context.conv_state_vec.clear();
    forward_context.ssm_state_vec.clear();
    if (nullptr == cache_config) {
        return;
    }

    const backends::AttentionBackend attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;

    auto cache_vectors = infinilm::models::qwen3_next::qwen3_next_allocate_cache_tensors(
        cache_config, model_config_, attention_backend);
    forward_context.kv_cache_vec = std::move(cache_vectors.kv_cache_tensors);
    forward_context.conv_state_vec = std::move(cache_vectors.conv_state_tensors);
    forward_context.ssm_state_vec = std::move(cache_vectors.ssm_state_tensors);
}

} // namespace infinilm::models::qwen3_5
