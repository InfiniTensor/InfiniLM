#pragma once

#include "../../layers/linear/linear.hpp"

#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include <runtime/tensor.hpp>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace infinilm::models::kimi_k3 {

class KimiK3VisionPatchProjection : public infinilm::weights::ModelWeights {
public:
    KimiK3VisionPatchProjection(
    infinilm::runtime::Runtime &runtime,
    size_t hidden_size,
                                size_t patch_size,
    const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values) const;

protected:
    size_t patch_size_{0};
    infinilm::weights::Weight weight_;
};

class KimiK3VisionPosEmbed : public infinilm::weights::ModelWeights {
public:
    KimiK3VisionPosEmbed(
    infinilm::runtime::Runtime &runtime,
    size_t height,
                         size_t width,
                         size_t hidden_size,
    const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, size_t grid_h, size_t grid_w) const;

protected:
    size_t height_{0};
    size_t width_{0};
    size_t hidden_size_{0};
    infinilm::weights::Weight weight_;
};

class KimiK3VisionPatchEmbed : public infinilm::weights::ModelWeights {
public:
    KimiK3VisionPatchEmbed(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values,
                               const infinilm::runtime::Tensor &grid_thw) const;

protected:
    std::shared_ptr<KimiK3VisionPatchProjection> proj_;
    std::shared_ptr<KimiK3VisionPosEmbed> pos_emb_;
};

class KimiK3VisionMLP : public infinilm::weights::ModelWeights {
public:
    KimiK3VisionMLP(
    infinilm::runtime::Runtime &runtime,
    size_t hidden_size,
                    size_t intermediate_size,
    const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

protected:
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> fc0_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> fc1_;
};

class KimiK3VisionBlock : public infinilm::weights::ModelWeights {
public:
    KimiK3VisionBlock(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb,
    const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                               const infinilm::runtime::Tensor &row_positions,
                               const infinilm::runtime::Tensor &col_positions) const;

protected:
    size_t hidden_size_{0};
    size_t num_heads_{0};
    size_t head_dim_{0};
    float scale_{1.0f};
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb_;

    std::shared_ptr<infinilm::layers::RMSNorm> norm0_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm1_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wqkv_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wo_;
    std::shared_ptr<KimiK3VisionMLP> mlp_;
};

class KimiK3VisionEncoder : public infinilm::weights::ModelWeights {
public:
    KimiK3VisionEncoder(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                               const infinilm::runtime::Tensor &grid_thw) const;

protected:
    std::vector<std::shared_ptr<KimiK3VisionBlock>> blocks_;
    std::shared_ptr<infinilm::layers::RMSNorm> final_layernorm_;
};

class KimiK3VisionTower : public infinilm::weights::ModelWeights {
public:
    KimiK3VisionTower(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values,
                               const infinilm::runtime::Tensor &grid_thw) const;

protected:
    size_t merge_height_{2};
    size_t merge_width_{2};
    size_t hidden_size_{0};
    std::shared_ptr<KimiK3VisionPatchEmbed> patch_embed_;
    std::shared_ptr<KimiK3VisionEncoder> encoder_;
};

class KimiK3Projector : public infinilm::weights::ModelWeights {
public:
    KimiK3Projector(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &vision_features) const;

protected:
    size_t merged_hidden_size_{0};
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> linear_0_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> linear_2_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_norm_;
};

} // namespace infinilm::models::kimi_k3
