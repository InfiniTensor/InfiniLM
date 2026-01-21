#pragma once

#include "minicpmv_config.hpp"

#include "infinicore/nn/layernorm.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <optional>
#include <vector>

namespace infinilm::models::minicpmv {

class ResamplerAttention : public infinicore::nn::Module {
public:
    ResamplerAttention(size_t embed_dim,
                       size_t num_heads,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &query,
                               const infinicore::Tensor &key,
                               const infinicore::Tensor &value) const;

private:
    size_t embed_dim_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;

    INFINICORE_NN_PARAMETER(in_proj_weight);
    INFINICORE_NN_PARAMETER(in_proj_bias);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, out_proj);
};

class Resampler : public infinicore::nn::Module {
public:
    Resampler(size_t num_queries,
              size_t embed_dim,
              size_t num_heads,
              size_t kv_dim,
              const infinicore::DataType &dtype,
              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &x,
                               const std::optional<infinicore::Tensor> &tgt_sizes) const;

private:
    size_t num_queries_;
    size_t embed_dim_;
    size_t num_heads_;
    size_t kv_dim_;
    bool use_kv_proj_;

    INFINICORE_NN_PARAMETER(query);
    INFINICORE_NN_PARAMETER(proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, kv_proj);
    INFINICORE_NN_MODULE(ResamplerAttention, attn);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln_q);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln_kv);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln_post);
};

} // namespace infinilm::models::minicpmv
