#pragma once

#include "infinicore/nn/embedding.hpp"
#include "infinicore/tensor.hpp"

#include <tuple>
#include <vector>

namespace infinilm::models::qwen3_vl {

class Qwen3VLPositionBuilder {
public:
    Qwen3VLPositionBuilder(size_t hidden_size,
                           size_t spatial_merge_size,
                           size_t num_grid_per_side,
                           size_t num_heads,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device);

    infinicore::Tensor position_embeddings(const infinicore::Tensor &image_grid_thw,
                                           const infinicore::nn::Embedding &pos_embed) const;

    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
    rotary_embeddings(const infinicore::Tensor &image_grid_thw) const;

private:
    infinicore::Tensor values_to_device_(const std::vector<float> &values,
                                         const infinicore::Shape &shape) const;

    size_t hidden_size_;
    size_t spatial_merge_size_;
    size_t num_grid_per_side_;
    size_t num_heads_;
    infinicore::DataType dtype_;
    infinicore::Device device_;
};

} // namespace infinilm::models::qwen3_vl
