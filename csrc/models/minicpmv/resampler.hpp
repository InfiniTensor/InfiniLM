#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/normalization/layer_norm.hpp"
#include "../../weights/weights.hpp"
#include "runtime/tensor.hpp"

#include <optional>
#include <vector>

namespace infinilm::models::minicpmv {

class ResamplerAttention : public infinilm::weights::ModelWeights {
public:
    ResamplerAttention(infinilm::runtime::Runtime &runtime, size_t embed_dim,
                       size_t num_heads,
                       const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &query,
                                      const infinilm::runtime::Tensor &key,
                                      const infinilm::runtime::Tensor &value) const;

private:
    size_t embed_dim_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;

    infinilm::weights::Weight in_proj_weight_;
    infinilm::weights::Weight in_proj_bias_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> out_proj_;
};

class Resampler : public infinilm::weights::ModelWeights {
public:
    Resampler(infinilm::runtime::Runtime &runtime, size_t num_queries,
              size_t embed_dim,
              size_t num_heads,
              size_t kv_dim,
              size_t image_size,
              size_t patch_size,
              const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &x,
                                      const infinilm::runtime::Tensor &tgt_sizes) const;

private:
    size_t num_queries_;
    size_t embed_dim_;
    size_t num_heads_;
    size_t kv_dim_;
    size_t image_size_;
    size_t patch_size_;
    bool use_kv_proj_;

    infinilm::weights::Weight query_;
    infinilm::weights::Weight proj_;
    infinilm::runtime::Tensor embedding_table_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> kv_proj_;
    std::shared_ptr<ResamplerAttention> attn_;
    std::shared_ptr<infinilm::layers::LayerNorm> ln_q_;
    std::shared_ptr<infinilm::layers::LayerNorm> ln_kv_;
    std::shared_ptr<infinilm::layers::LayerNorm> ln_post_;
};

} // namespace infinilm::models::minicpmv
