#pragma once

#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"
#include <nlohmann/json.hpp>

namespace infinilm::models::ernie4_5_moe_vl {

class Ernie4_5VisionPatchEmbed : public infinicore::nn::Module {
public:
    Ernie4_5VisionPatchEmbed(const nlohmann::json &vision_config,
                             const infinicore::DataType &dtype,
                             const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Linear, proj);
};

class Ernie4_5VisionMLP : public infinicore::nn::Module {
public:
    Ernie4_5VisionMLP(size_t hidden_size,
                      double mlp_ratio,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Linear, fc1);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, fc2);
};

class Ernie4_5VisionAttention : public infinicore::nn::Module {
public:
    Ernie4_5VisionAttention(size_t hidden_size,
                            size_t num_heads,
                            const infinicore::DataType &dtype,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &rotary_pos_ids,
                               const infinicore::Tensor &grid_thw) const;

private:
    size_t hidden_size_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;

    INFINICORE_NN_MODULE(infinicore::nn::Linear, qkv);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, proj);
};

class Ernie4_5VisionBlock : public infinicore::nn::Module {
public:
    Ernie4_5VisionBlock(const nlohmann::json &vision_config,
                        const infinicore::DataType &dtype,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &rotary_pos_ids,
                               const infinicore::Tensor &grid_thw) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm1);
    INFINICORE_NN_MODULE(Ernie4_5VisionAttention, attn);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm2);
    INFINICORE_NN_MODULE(Ernie4_5VisionMLP, mlp);
};

class Ernie4_5VisionTransformer : public infinicore::nn::Module {
public:
    Ernie4_5VisionTransformer(const nlohmann::json &vision_config,
                              double norm_eps,
                              const infinicore::DataType &dtype,
                              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const infinicore::Tensor &grid_thw) const;

private:
    size_t spatial_merge_size_{2};

    infinicore::Tensor build_rotary_pos_ids(const infinicore::Tensor &grid_thw,
                                            size_t seq_len) const;

    INFINICORE_NN_MODULE(Ernie4_5VisionPatchEmbed, patch_embed);
    INFINICORE_NN_MODULE_VEC(Ernie4_5VisionBlock, blocks);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln);
};

class Ernie4_5VariableResolutionResampler : public infinicore::nn::Module {
public:
    Ernie4_5VariableResolutionResampler(const nlohmann::json &config,
                                        const infinicore::DataType &dtype,
                                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &grid_thw) const;

private:
    infinicore::Tensor spatial_forward(const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor temporal_forward(const infinicore::Tensor &hidden_states,
                                        const infinicore::Tensor &grid_thw) const;

    size_t in_dim_;
    size_t out_dim_;
    size_t spatial_conv_size_;
    size_t temporal_conv_size_;
    bool use_temporal_conv_;

    std::shared_ptr<infinicore::nn::Linear> spatial_linear0_;
    std::shared_ptr<infinicore::nn::Linear> spatial_linear2_;
    std::shared_ptr<infinicore::nn::LayerNorm> spatial_linear3_;
    std::shared_ptr<infinicore::nn::Linear> temporal_linear0_;
    std::shared_ptr<infinicore::nn::Linear> temporal_linear2_;
    std::shared_ptr<infinicore::nn::LayerNorm> temporal_linear3_;
    std::shared_ptr<infinicore::nn::Linear> mlp_;
    std::shared_ptr<infinicore::nn::RMSNorm> after_norm_;
};

} // namespace infinilm::models::ernie4_5_moe_vl
