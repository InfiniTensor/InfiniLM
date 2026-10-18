#pragma once
#include "../../layers/normalization/rmsnorm.hpp"

#include "../../layers/common_modules.hpp"
#include "../../layers/normalization/layer_norm.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include "../../weights/weights.hpp"
#include "runtime/tensor.hpp"
#include <memory>
#include <nlohmann/json.hpp>

namespace infinilm::models::ernie4_5_vl {
class Ernie45VisionLayerNorm : public infinilm::weights::ModelWeights {
public:
    Ernie45VisionLayerNorm(infinilm::runtime::Runtime &runtime,
                           size_t normalized_shape,
                           double eps,
                           const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states) const;

private:
    infinilm::weights::Weight weight_;
    infinilm::weights::Weight bias_;
    double eps_{1e-6};
};

class Ernie45VisionPatchEmbed : public infinilm::weights::ModelWeights {
public:
    Ernie45VisionPatchEmbed(infinilm::runtime::Runtime &runtime,
                            const nlohmann::json &config,
                            const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states) const;

private:
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> proj_;
};

class Ernie45VisionAttention : public infinilm::weights::ModelWeights {
public:
    Ernie45VisionAttention(infinilm::runtime::Runtime &runtime,
                           const nlohmann::json &config,
                           const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      const infinilm::runtime::Tensor &row_position_ids,
                                      const infinilm::runtime::Tensor &col_position_ids) const;

private:
    size_t hidden_size_{1280};
    size_t num_heads_{16};
    size_t head_dim_{80};
    float scale_{1.0f};

    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> qkv_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> proj_;
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb_;
};

class Ernie45VisionMLP : public infinilm::weights::ModelWeights {
public:
    Ernie45VisionMLP(infinilm::runtime::Runtime &runtime,
                     const nlohmann::json &config,
                     const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states) const;

private:
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> fc1_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> fc2_;
};

class Ernie45VisionBlock : public infinilm::weights::ModelWeights {
public:
    Ernie45VisionBlock(infinilm::runtime::Runtime &runtime,
                       const nlohmann::json &config,
                       const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      const infinilm::runtime::Tensor &row_position_ids,
                                      const infinilm::runtime::Tensor &col_position_ids) const;

private:
    std::shared_ptr<Ernie45VisionLayerNorm> norm1_;
    std::shared_ptr<Ernie45VisionLayerNorm> norm2_;
    std::shared_ptr<Ernie45VisionAttention> attn_;
    std::shared_ptr<Ernie45VisionMLP> mlp_;
};

class Ernie45VisionModel : public infinilm::weights::ModelWeights {
public:
    Ernie45VisionModel(infinilm::runtime::Runtime &runtime,
                       const nlohmann::json &config,
                       const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &pixel_values,
                                      const infinilm::runtime::Tensor &grid_thw) const;

private:
    infinilm::runtime::Tensor build_rotary_position_ids(infinilm::runtime::Runtime &runtime,
                                                        const infinilm::runtime::Tensor &grid_thw) const;

    size_t spatial_merge_size_{2};

    std::shared_ptr<Ernie45VisionPatchEmbed> patch_embed_;
    std::vector<std::shared_ptr<Ernie45VisionBlock>> blocks_;
    std::shared_ptr<Ernie45VisionLayerNorm> ln_;
};

class Ernie45ResamplerModel : public infinilm::weights::ModelWeights {
public:
    Ernie45ResamplerModel(infinilm::runtime::Runtime &runtime,
                          const nlohmann::json &config,
                          const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &image_features,
                                      const infinilm::runtime::Tensor &grid_thw) const;

private:
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> spatial_linear_0_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> spatial_linear_2_;
    std::shared_ptr<Ernie45VisionLayerNorm> spatial_linear_3_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> temporal_linear_0_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> temporal_linear_2_;
    std::shared_ptr<Ernie45VisionLayerNorm> temporal_linear_3_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> mlp_;
    std::shared_ptr<infinilm::layers::RMSNorm> after_norm_;
    bool use_temporal_conv_{true};
    size_t spatial_conv_size_{2};
    size_t temporal_conv_size_{2};
};

} // namespace infinilm::models::ernie4_5_vl
