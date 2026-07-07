#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/common_modules.hpp"

#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_5 {

class Qwen35VisionPatchProj : public infinicore::nn::Module {
public:
    Qwen35VisionPatchProj(size_t in_channels,
                          size_t hidden_size,
                          size_t temporal_patch_size,
                          size_t patch_size,
                          const infinicore::DataType &dtype,
                          const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    size_t in_channels_;
    size_t hidden_size_;
    size_t temporal_patch_size_;
    size_t patch_size_;

    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);
};

class Qwen35VisionPatchEmbed : public infinicore::nn::Module {
public:
    Qwen35VisionPatchEmbed(const nlohmann::json &config,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(Qwen35VisionPatchProj, proj);
};

class Qwen35VisionAttention : public infinicore::nn::Module {
public:
    Qwen35VisionAttention(const nlohmann::json &config,
                          const infinicore::DataType &dtype,
                          const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &row_position_ids,
                               const infinicore::Tensor &col_position_ids) const;

private:
    size_t hidden_size_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;

    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, qkv);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, proj);
    INFINICORE_NN_MODULE(infinicore::nn::RoPE, rotary_emb);
};

class Qwen35VisionMLP : public infinicore::nn::Module {
public:
    Qwen35VisionMLP(const nlohmann::json &config,
                    const infinicore::DataType &dtype,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    std::string activation_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_fc1);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_fc2);
};

class Qwen35VisionBlock : public infinicore::nn::Module {
public:
    Qwen35VisionBlock(const nlohmann::json &config,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &row_position_ids,
                               const infinicore::Tensor &col_position_ids) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm1);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm2);
    INFINICORE_NN_MODULE(Qwen35VisionAttention, attn);
    INFINICORE_NN_MODULE(Qwen35VisionMLP, mlp);
};

class Qwen35VisionPatchMerger : public infinicore::nn::Module {
public:
    Qwen35VisionPatchMerger(const nlohmann::json &config,
                            const infinicore::DataType &dtype,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    size_t hidden_size_;
    size_t merged_size_;
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_fc1);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_fc2);
};

class Qwen35VisionModel : public infinicore::nn::Module {
public:
    Qwen35VisionModel(const nlohmann::json &config,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const infinicore::Tensor &image_grid_thw) const;

private:
    infinicore::Tensor fast_pos_embed_interpolate(const infinicore::Tensor &image_grid_thw) const;
    infinicore::Tensor build_rotary_position_ids(const infinicore::Tensor &image_grid_thw) const;

    size_t hidden_size_;
    size_t num_heads_;
    size_t head_dim_;
    size_t spatial_merge_size_;
    size_t num_grid_per_side_;

    INFINICORE_NN_MODULE(Qwen35VisionPatchEmbed, patch_embed);
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, pos_embed);
    INFINICORE_NN_MODULE_VEC(Qwen35VisionBlock, blocks);
    INFINICORE_NN_MODULE(Qwen35VisionPatchMerger, merger);
};

} // namespace infinilm::models::qwen3_5
