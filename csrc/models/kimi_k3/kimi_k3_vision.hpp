#pragma once

#include "../../layers/linear/linear.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/nn/rope.hpp>
#include <infinicore/tensor.hpp>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace infinilm::models::kimi_k3 {

class KimiK3VisionPatchProjection : public infinicore::nn::Module {
public:
    KimiK3VisionPatchProjection(size_t hidden_size,
                                size_t patch_size,
                                const infinicore::DataType &dtype,
                                const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &pixel_values) const;

protected:
    size_t patch_size_{0};
    INFINICORE_NN_PARAMETER(weight);
};

class KimiK3VisionPosEmbed : public infinicore::nn::Module {
public:
    KimiK3VisionPosEmbed(size_t height,
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

class KimiK3VisionPatchEmbed : public infinicore::nn::Module {
public:
    KimiK3VisionPatchEmbed(const nlohmann::json &config,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const infinicore::Tensor &grid_thw) const;

protected:
    INFINICORE_NN_MODULE(KimiK3VisionPatchProjection, proj);
    INFINICORE_NN_MODULE(KimiK3VisionPosEmbed, pos_emb);
};

class KimiK3VisionMLP : public infinicore::nn::Module {
public:
    KimiK3VisionMLP(size_t hidden_size,
                    size_t intermediate_size,
                    const infinicore::DataType &dtype,
                    const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, fc0);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, fc1);
};

class KimiK3VisionBlock : public infinicore::nn::Module {
public:
    KimiK3VisionBlock(const nlohmann::json &config,
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

    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm0);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm1);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, wqkv);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, wo);
    INFINICORE_NN_MODULE(KimiK3VisionMLP, mlp);
};

class KimiK3VisionEncoder : public infinicore::nn::Module {
public:
    KimiK3VisionEncoder(const nlohmann::json &config,
                        const infinicore::DataType &dtype,
                        const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &grid_thw) const;

protected:
    INFINICORE_NN_MODULE_VEC(KimiK3VisionBlock, blocks);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, final_layernorm);
};

class KimiK3VisionTower : public infinicore::nn::Module {
public:
    KimiK3VisionTower(const nlohmann::json &config,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const infinicore::Tensor &grid_thw) const;

protected:
    size_t merge_height_{2};
    size_t merge_width_{2};
    size_t hidden_size_{0};
    INFINICORE_NN_MODULE(KimiK3VisionPatchEmbed, patch_embed);
    INFINICORE_NN_MODULE(KimiK3VisionEncoder, encoder);
};

class KimiK3Projector : public infinicore::nn::Module {
public:
    KimiK3Projector(const nlohmann::json &config,
                    const infinicore::DataType &dtype,
                    const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &vision_features) const;

protected:
    size_t merged_hidden_size_{0};
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_0);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, linear_2);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_norm);
};

} // namespace infinilm::models::kimi_k3
