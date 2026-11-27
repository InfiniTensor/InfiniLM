#pragma once

#include "../../layers/linear/linear.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"
#include "gemma4_vision.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

namespace infinilm::models::gemma4 {

class Gemma4AudioSubSampleLayer : public infinilm::weights::ModelWeights {
public:
    Gemma4AudioSubSampleLayer(infinilm::runtime::Runtime &runtime,
                              size_t in_channels,
                              size_t out_channels,
                              double norm_eps,
                              infinilm::runtime::DataType dtype);

    // Input: [1, C_in, T, F] -> Output: [1, C_out, ceil(T/2), ceil(F/2)]
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &x) const;

private:
    size_t in_channels_{1};
    size_t out_channels_{128};
    double norm_eps_{1e-6};
    infinilm::weights::Weight conv_weight_; // [out_channels, in_channels, 3, 3]
    infinilm::weights::Weight norm_weight_; // [out_channels]
    infinilm::runtime::Tensor norm_zero_bias_;
};

class Gemma4AudioSubSampleConvProjection : public infinilm::weights::ModelWeights {
public:
    Gemma4AudioSubSampleConvProjection(infinilm::runtime::Runtime &runtime,
                                       const nlohmann::json &audio_cfg,
                                       infinilm::runtime::DataType dtype);

    // Input: [T, F] (e.g. [T, 128]) -> Output: [T_sub, hidden_size]
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &input_features) const;

private:
    std::shared_ptr<Gemma4AudioSubSampleLayer> layer0_;
    std::shared_ptr<Gemma4AudioSubSampleLayer> layer1_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> input_proj_linear_;
};

class Gemma4AudioFeedForward : public infinilm::weights::ModelWeights {
public:
    Gemma4AudioFeedForward(infinilm::runtime::Runtime &runtime,
                           const nlohmann::json &audio_cfg,
                           infinilm::runtime::DataType dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &x) const;

private:
    double residual_weight_{0.5};
    double gradient_clipping_{1e10};
    std::shared_ptr<Gemma4ClippableLinear> ffw_layer_1_;
    std::shared_ptr<Gemma4ClippableLinear> ffw_layer_2_;
    std::shared_ptr<infinilm::layers::RMSNorm> pre_layer_norm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_layer_norm_;
};

class Gemma4AudioLightConv1d : public infinilm::weights::ModelWeights {
public:
    Gemma4AudioLightConv1d(infinilm::runtime::Runtime &runtime,
                           const nlohmann::json &audio_cfg,
                           infinilm::runtime::DataType dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &x) const;

private:
    size_t hidden_size_{1024};
    size_t kernel_size_{5};
    std::shared_ptr<Gemma4ClippableLinear> linear_start_;
    std::shared_ptr<Gemma4ClippableLinear> linear_end_;
    infinilm::weights::Weight depthwise_conv1d_weight_; // [hidden_size, 1, kernel_size]
    std::shared_ptr<infinilm::layers::RMSNorm> pre_layer_norm_;
    std::shared_ptr<infinilm::layers::RMSNorm> conv_norm_;
};

class Gemma4AudioAttention : public infinilm::weights::ModelWeights {
public:
    Gemma4AudioAttention(infinilm::runtime::Runtime &runtime,
                         const nlohmann::json &audio_cfg,
                         infinilm::runtime::DataType dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &x,
                                      const infinilm::runtime::Tensor &rel_pos_embed) const;

private:
    size_t hidden_size_{1024};
    size_t num_heads_{8};
    size_t head_dim_{128};
    size_t chunk_size_{12};
    size_t max_past_horizon_{12};
    size_t max_future_horizon_{0};
    size_t context_size_{24};
    double softcap_{50.0};
    double invalid_logits_value_{-1e9};
    double q_scale_{1.0};
    double k_scale_{1.0};

    std::shared_ptr<Gemma4ClippableLinear> q_proj_;
    std::shared_ptr<Gemma4ClippableLinear> k_proj_;
    std::shared_ptr<Gemma4ClippableLinear> v_proj_;
    std::shared_ptr<Gemma4ClippableLinear> post_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> relative_k_proj_;
    infinilm::weights::Weight per_dim_scale_; // [head_dim]
};

class Gemma4AudioLayer : public infinilm::weights::ModelWeights {
public:
    Gemma4AudioLayer(infinilm::runtime::Runtime &runtime,
                     const nlohmann::json &audio_cfg,
                     infinilm::runtime::DataType dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &x,
                                      const infinilm::runtime::Tensor &rel_pos_embed) const;

private:
    std::shared_ptr<Gemma4AudioFeedForward> feed_forward1_;
    std::shared_ptr<Gemma4AudioFeedForward> feed_forward2_;
    std::shared_ptr<Gemma4AudioAttention> self_attn_;
    std::shared_ptr<Gemma4AudioLightConv1d> lconv1d_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_pre_attn_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_post_attn_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_out_;
};

class Gemma4AudioModel : public infinilm::weights::ModelWeights {
public:
    Gemma4AudioModel(infinilm::runtime::Runtime &runtime,
                     const nlohmann::json &audio_cfg,
                     infinilm::runtime::DataType dtype);

    // Input: [T_mel, 128] -> Output: [T_out, output_proj_dims]
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &input_features) const;

private:
    size_t hidden_size_{1024};
    size_t context_size_{24};
    infinilm::runtime::DataType dtype_{infinilm::runtime::DataType::kBFloat16};
    infinilm::runtime::Tensor rel_pos_embed_; // [context_size/2 + 1, hidden_size]

    std::shared_ptr<Gemma4AudioSubSampleConvProjection> subsample_conv_projection_;
    std::vector<std::shared_ptr<Gemma4AudioLayer>> layers_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> output_proj_;
};

} // namespace infinilm::models::gemma4
