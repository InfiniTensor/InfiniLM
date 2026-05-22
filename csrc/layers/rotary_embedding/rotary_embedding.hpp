#pragma once

#include "../../config/model_config.hpp"
#include "infinicore/nn/rope.hpp"
#include <memory>

namespace infinilm::layers::rotary_embedding {

// Compute the actual number of dimensions involved in rotary position embedding.
// For partial rotation, the dimension is clamped to [2, head_dim] and must be even.
size_t get_rotary_dim(size_t head_dim, double partial_rotary_factor);

std::shared_ptr<infinicore::nn::RoPE> get_rope(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                                               const infinicore::Device &device,
                                               infinicore::nn::RoPE::Algo algo = infinicore::nn::RoPE::Algo::GPT_NEOX);

} // namespace infinilm::layers::rotary_embedding
