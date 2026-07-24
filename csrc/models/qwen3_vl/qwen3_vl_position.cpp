#include "qwen3_vl_position.hpp"

#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace infinilm::models::qwen3_vl {
namespace {

size_t grid_num_patches(const int64_t *grid, size_t num_grids) {
    size_t total_patches = 0;
    for (size_t i = 0; i < num_grids; ++i) {
        total_patches += static_cast<size_t>(grid[i * 3])
                       * static_cast<size_t>(grid[i * 3 + 1])
                       * static_cast<size_t>(grid[i * 3 + 2]);
    }
    return total_patches;
}

void validate_grid(size_t h, size_t w, size_t spatial_merge_size) {
    if (spatial_merge_size == 0 || h % spatial_merge_size != 0 || w % spatial_merge_size != 0) {
        throw std::runtime_error("Qwen3VLPositionBuilder: image grid must be divisible by spatial_merge_size");
    }
}

float load_scalar(const std::byte *ptr, infinicore::DataType dtype) {
    switch (dtype) {
    case infinicore::DataType::F32:
        return *reinterpret_cast<const float *>(ptr);
    case infinicore::DataType::F16:
        return f16_to_f32(*reinterpret_cast<const uint16_t *>(ptr));
    case infinicore::DataType::BF16:
        return bf16_to_f32(*reinterpret_cast<const uint16_t *>(ptr));
    default:
        throw std::runtime_error("Qwen3VLPositionBuilder: unsupported pos_embed dtype");
    }
}

} // namespace

Qwen3VLPositionBuilder::Qwen3VLPositionBuilder(size_t hidden_size,
                                               size_t spatial_merge_size,
                                               size_t num_grid_per_side,
                                               size_t num_heads,
                                               const infinicore::DataType &dtype,
                                               const infinicore::Device &device)
    : hidden_size_(hidden_size),
      spatial_merge_size_(spatial_merge_size),
      num_grid_per_side_(num_grid_per_side),
      num_heads_(num_heads),
      dtype_(dtype),
      device_(device) {}

infinicore::Tensor Qwen3VLPositionBuilder::values_to_device_(const std::vector<float> &values,
                                                             const infinicore::Shape &shape) const {
    if (dtype_ == infinicore::DataType::F32) {
        auto cpu = infinicore::Tensor::from_blob(const_cast<float *>(values.data()), shape, dtype_, infinicore::Device::cpu());
        return cpu->to(device_);
    }

    std::vector<uint16_t> packed(values.size());
    if (dtype_ == infinicore::DataType::BF16) {
        for (size_t i = 0; i < values.size(); ++i) {
            packed[i] = f32_to_bf16(values[i]);
        }
    } else if (dtype_ == infinicore::DataType::F16) {
        for (size_t i = 0; i < values.size(); ++i) {
            packed[i] = f32_to_f16(values[i]);
        }
    } else {
        throw std::runtime_error("Qwen3VLPositionBuilder: unsupported dtype for generated position tables");
    }

    auto cpu = infinicore::Tensor::from_blob(packed.data(), shape, dtype_, infinicore::Device::cpu());
    return cpu->to(device_);
}

infinicore::Tensor Qwen3VLPositionBuilder::position_embeddings(const infinicore::Tensor &image_grid_thw,
                                                               const infinicore::nn::Embedding &pos_embed) const {
    auto grid_cpu = image_grid_thw->to(infinicore::Device::cpu());
    auto weight_cpu = pos_embed.weight()->to(infinicore::Device::cpu());
    const int64_t *grid = reinterpret_cast<const int64_t *>(grid_cpu->data());
    const size_t num_grids = grid_cpu->size(0);
    const size_t total_patches = grid_num_patches(grid, num_grids);
    const size_t num_positions = weight_cpu->size(0);
    const size_t hidden_size = weight_cpu->size(1);
    const size_t elem_size = weight_cpu->element_size();
    const auto dtype = weight_cpu->dtype();
    const auto *weight = weight_cpu->data();
    std::vector<float> values(total_patches * hidden_size);

    size_t offset = 0;
    for (size_t g = 0; g < num_grids; ++g) {
        const size_t t = static_cast<size_t>(grid[g * 3]);
        const size_t h = static_cast<size_t>(grid[g * 3 + 1]);
        const size_t w = static_cast<size_t>(grid[g * 3 + 2]);
        validate_grid(h, w, spatial_merge_size_);

        for (size_t token = 0; token < t * h * w; ++token) {
            const size_t frame_offset = token % (h * w);
            const size_t merged_idx = frame_offset / (spatial_merge_size_ * spatial_merge_size_);
            const size_t intra = frame_offset % (spatial_merge_size_ * spatial_merge_size_);
            const size_t merged_h = merged_idx / (w / spatial_merge_size_);
            const size_t merged_w = merged_idx % (w / spatial_merge_size_);
            const size_t ph = merged_h * spatial_merge_size_ + intra / spatial_merge_size_;
            const size_t pw = merged_w * spatial_merge_size_ + intra % spatial_merge_size_;

            const float h_pos = h > 1 ? static_cast<float>(ph) * static_cast<float>(num_grid_per_side_ - 1) / static_cast<float>(h - 1) : 0.0f;
            const float w_pos = w > 1 ? static_cast<float>(pw) * static_cast<float>(num_grid_per_side_ - 1) / static_cast<float>(w - 1) : 0.0f;
            const size_t h_floor = static_cast<size_t>(std::floor(h_pos));
            const size_t w_floor = static_cast<size_t>(std::floor(w_pos));
            const size_t h_ceil = std::min(h_floor + 1, num_grid_per_side_ - 1);
            const size_t w_ceil = std::min(w_floor + 1, num_grid_per_side_ - 1);
            const float dh = h_pos - static_cast<float>(h_floor);
            const float dw = w_pos - static_cast<float>(w_floor);

            const size_t pos_ids[4] = {
                h_floor * num_grid_per_side_ + w_floor,
                h_floor * num_grid_per_side_ + w_ceil,
                h_ceil * num_grid_per_side_ + w_floor,
                h_ceil * num_grid_per_side_ + w_ceil,
            };
            const float weights[4] = {
                (1.0f - dh) * (1.0f - dw),
                (1.0f - dh) * dw,
                dh * (1.0f - dw),
                dh * dw,
            };

            for (size_t k = 0; k < 4; ++k) {
                if (pos_ids[k] >= num_positions) {
                    throw std::runtime_error("Qwen3VLPositionBuilder: generated position id is out of range");
                }
            }

            for (size_t d = 0; d < hidden_size; ++d) {
                float value = 0.0f;
                for (size_t k = 0; k < 4; ++k) {
                    const auto *src = weight + (pos_ids[k] * hidden_size + d) * elem_size;
                    value += weights[k] * load_scalar(src, dtype);
                }
                values[offset * hidden_size + d] = value;
            }
            ++offset;
        }
    }

    return values_to_device_(values, {total_patches, hidden_size});
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
Qwen3VLPositionBuilder::rotary_embeddings(const infinicore::Tensor &image_grid_thw) const {
    auto grid_cpu = image_grid_thw->to(infinicore::Device::cpu());
    const int64_t *grid = reinterpret_cast<const int64_t *>(grid_cpu->data());
    const size_t num_grids = grid_cpu->size(0);
    const size_t total_patches = grid_num_patches(grid, num_grids);

    auto pos_cpu = infinicore::Tensor::empty({total_patches}, infinicore::DataType::I64, infinicore::Device::cpu());
    auto *pos = reinterpret_cast<int64_t *>(pos_cpu->data());
    for (size_t i = 0; i < total_patches; ++i) {
        pos[i] = static_cast<int64_t>(i);
    }

    const size_t head_dim = hidden_size_ / num_heads_;
    const size_t half_dim = head_dim / 2;
    const size_t axis_dim = half_dim / 2;
    std::vector<float> sin_values(total_patches * half_dim);
    std::vector<float> cos_values(total_patches * half_dim);

    size_t offset = 0;
    for (size_t g = 0; g < num_grids; ++g) {
        const size_t t = static_cast<size_t>(grid[g * 3]);
        const size_t h = static_cast<size_t>(grid[g * 3 + 1]);
        const size_t w = static_cast<size_t>(grid[g * 3 + 2]);
        validate_grid(h, w, spatial_merge_size_);

        for (size_t token = 0; token < t * h * w; ++token) {
            const size_t frame_offset = token % (h * w);
            const size_t merged_idx = frame_offset / (spatial_merge_size_ * spatial_merge_size_);
            const size_t intra = frame_offset % (spatial_merge_size_ * spatial_merge_size_);
            const size_t merged_h = merged_idx / (w / spatial_merge_size_);
            const size_t merged_w = merged_idx % (w / spatial_merge_size_);
            const size_t py = merged_h * spatial_merge_size_ + intra / spatial_merge_size_;
            const size_t px = merged_w * spatial_merge_size_ + intra % spatial_merge_size_;

            for (size_t d = 0; d < axis_dim; ++d) {
                const float inv_freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * d) / static_cast<float>(half_dim));
                const float ay = static_cast<float>(py) * inv_freq;
                const float ax = static_cast<float>(px) * inv_freq;
                sin_values[offset * half_dim + d] = std::sin(ay);
                cos_values[offset * half_dim + d] = std::cos(ay);
                sin_values[offset * half_dim + axis_dim + d] = std::sin(ax);
                cos_values[offset * half_dim + axis_dim + d] = std::cos(ax);
            }
            ++offset;
        }
    }

    return {pos_cpu->to(device_),
            values_to_device_(sin_values, {total_patches, half_dim}),
            values_to_device_(cos_values, {total_patches, half_dim})};
}

} // namespace infinilm::models::qwen3_vl
