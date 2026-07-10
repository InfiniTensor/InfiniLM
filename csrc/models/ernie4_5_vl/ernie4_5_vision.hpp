#pragma once

#include "../../layers/common_modules.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <nlohmann/json.hpp>
#include <memory>

namespace infinilm::models::ernie4_5_vl {
class Ernie45VisionLayerNorm : public infinicore::nn::Module {
public:
    Ernie45VisionLayerNorm(size_t normalized_shape,
                           double eps,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);
    double eps_{1e-6};
};

class Ernie45VisionPatchEmbed : public infinicore::nn::Module {
public:
    Ernie45VisionPatchEmbed(const nlohmann::json &config,
                            const infinicore::DataType &dtype,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, proj);
};

class Ernie45VisionAttention : public infinicore::nn::Module {
public:
    Ernie45VisionAttention(const nlohmann::json &config,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, qkv);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, proj);
};

class Ernie45VisionMLP : public infinicore::nn::Module {
public:
    Ernie45VisionMLP(const nlohmann::json &config,
                     const infinicore::DataType &dtype,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, fc1);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, fc2);
};

class Ernie45VisionBlock : public infinicore::nn::Module {
public:
    Ernie45VisionBlock(const nlohmann::json &config,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(Ernie45VisionLayerNorm, norm1);
    INFINICORE_NN_MODULE(Ernie45VisionLayerNorm, norm2);
    INFINICORE_NN_MODULE(Ernie45VisionAttention, attn);
    INFINICORE_NN_MODULE(Ernie45VisionMLP, mlp);
};

class Ernie45VisionModel : public infinicore::nn::Module {
public:
    Ernie45VisionModel(const nlohmann::json &config,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values) const;

private:
    INFINICORE_NN_MODULE(Ernie45VisionPatchEmbed, patch_embed);
    INFINICORE_NN_MODULE_VEC(Ernie45VisionBlock, blocks);
    INFINICORE_NN_MODULE(Ernie45VisionLayerNorm, ln);
};

class Ernie45ResamplerModel : public infinicore::nn::Module {
public:
    Ernie45ResamplerModel(const nlohmann::json &config,
                          const infinicore::DataType &dtype,
                          const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &image_features) const;

private:
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> spatial_linear_0_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> spatial_linear_2_;
    std::shared_ptr<Ernie45VisionLayerNorm> spatial_linear_3_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> temporal_linear_0_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> temporal_linear_2_;
    std::shared_ptr<Ernie45VisionLayerNorm> temporal_linear_3_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, mlp);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, after_norm);
    bool use_temporal_conv_{true};
};

} // namespace infinilm::models::ernie4_5_vl

