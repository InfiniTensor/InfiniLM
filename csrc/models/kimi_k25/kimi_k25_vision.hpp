#pragma once

#include "../../layers/linear/linear.hpp"

#include <infinicore/nn/layer_norm.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rope.hpp>
#include <infinicore/tensor.hpp>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace infinilm::models::kimi_k25 {

class KimiK25VisionPatchProjection : public infinicore::nn::Module {
public:
    KimiK25VisionPatchProjection(size_t hidden_size,
                                 size_t patch_size,
                                 const infinicore::DataType &dtype,
                                 const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &pixel_values) const;

protected:
    size_t patch_size_{0};
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);
};

class KimiK25VisionPosEmbed : public infinicore::nn::Module {
public:
    KimiK25VisionPosEmbed(size_t height,
                          size_t width,
                          size_t hidden_size,
                          const infinicore::DataType &dtype,
                          const infinicore::Device &device);
    infinicore::Tensor forward(size_t grid_h, size_t grid_w) const;

protected:
    size_t height_{0};
    size_t width_{0};
    size_t hidden_size_{0};
    INFINICORE_NN_PARAMETER(weight);
};

class KimiK25VisionPatchEmbed : public infinicore::nn::Module {
public:
    KimiK25VisionPatchEmbed(const nlohmann::json &config,
                            const infinicore::DataType &dtype,
                            const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const infinicore::Tensor &grid_thw) const;

protected:
    INFINICORE_NN_MODULE(KimiK25VisionPatchProjection, proj);
    INFINICORE_NN_MODULE(KimiK25VisionPosEmbed, pos_emb);
};

class KimiK25VisionMLP : public infinicore::nn::Module {
public:
    KimiK25VisionMLP(size_t hidden_size,
                     size_t intermediate_size,
                     const infinicore::DataType &dtype,
                     const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, fc0);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, fc1);
};

class KimiK25VisionBlock : public infinicore::nn::Module {
public:
    KimiK25VisionBlock(const nlohmann::json &config,
                       std::shared_ptr<infinicore::nn::RoPE> rotary_emb,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &row_positions,
                               const infinicore::Tensor &col_positions) const;

protected:
    size_t hidden_size_{0};
    size_t num_heads_{0};
    size_t head_dim_{0};
    float scale_{1.0f};
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;

    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm0);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm1);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, wqkv);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, wo);
    INFINICORE_NN_MODULE(KimiK25VisionMLP, mlp);
};

class KimiK25VisionEncoder : public infinicore::nn::Module {
public:
    KimiK25VisionEncoder(const nlohmann::json &config,
                         const infinicore::DataType &dtype,
                         const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &grid_thw) const;

protected:
    INFINICORE_NN_MODULE_VEC(KimiK25VisionBlock, blocks);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, final_layernorm);
};

class KimiK25VisionTower : public infinicore::nn::Module {
public:
    KimiK25VisionTower(const nlohmann::json &config,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const infinicore::Tensor &grid_thw) const;

protected:
    size_t merge_height_{2};
    size_t merge_width_{2};
    size_t hidden_size_{0};
    INFINICORE_NN_MODULE(KimiK25VisionPatchEmbed, patch_embed);
    INFINICORE_NN_MODULE(KimiK25VisionEncoder, encoder);
};

class KimiK25Projector : public infinicore::nn::Module {
public:
    KimiK25Projector(const nlohmann::json &config,
                     const infinicore::DataType &dtype,
                     const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &vision_features) const;

protected:
    size_t merged_hidden_size_{0};
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, pre_norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_0);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_2);
};

} // namespace infinilm::models::kimi_k25
