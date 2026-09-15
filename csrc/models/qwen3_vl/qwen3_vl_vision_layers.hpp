#pragma once

#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <nlohmann/json.hpp>

namespace infinilm::models::qwen3_vl {

class Qwen3VLPatchProjection : public infinicore::nn::Module {
public:
    Qwen3VLPatchProjection(size_t out_features,
                           size_t in_channels,
                           size_t temporal_patch_size,
                           size_t patch_size,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values) const;

private:
    size_t patch_dim_;
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);
};

class Qwen3VLPatchEmbed : public infinicore::nn::Module {
public:
    Qwen3VLPatchEmbed(const nlohmann::json &config,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values) const;

private:
    INFINICORE_NN_MODULE(Qwen3VLPatchProjection, proj);
};

class Qwen3VLVisionMLP : public infinicore::nn::Module {
public:
    Qwen3VLVisionMLP(const nlohmann::json &config,
                     const infinicore::DataType &dtype,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_fc1);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_fc2);
};

class Qwen3VLVisionAttention : public infinicore::nn::Module {
public:
    Qwen3VLVisionAttention(const nlohmann::json &config,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &position_ids,
                               const infinicore::Tensor &sin_table,
                               const infinicore::Tensor &cos_table) const;

private:
    size_t hidden_size_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, qkv);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, proj);
};

class Qwen3VLVisionBlock : public infinicore::nn::Module {
public:
    Qwen3VLVisionBlock(const nlohmann::json &config,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &position_ids,
                               const infinicore::Tensor &sin_table,
                               const infinicore::Tensor &cos_table) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm1);
    INFINICORE_NN_MODULE(Qwen3VLVisionAttention, attn);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm2);
    INFINICORE_NN_MODULE(Qwen3VLVisionMLP, mlp);
};

class Qwen3VLPatchMerger : public infinicore::nn::Module {
public:
    Qwen3VLPatchMerger(const nlohmann::json &config,
                       bool use_postshuffle_norm,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    size_t hidden_size_;
    size_t merged_size_;
    bool use_postshuffle_norm_;
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_fc1);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_fc2);
};

} // namespace infinilm::models::qwen3_vl
