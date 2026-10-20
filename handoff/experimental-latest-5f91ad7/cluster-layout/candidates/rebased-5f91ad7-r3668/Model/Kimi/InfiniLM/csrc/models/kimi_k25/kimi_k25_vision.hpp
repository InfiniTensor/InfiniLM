#pragma once

#include "../../layers/linear/linear.hpp"

#include "../../layers/normalization/layer_norm.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include "../../weights/weights.hpp"
#include <nlohmann/json.hpp>
#include <runtime/tensor.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace infinilm::models::kimi_k25 {

class KimiK25VisionPatchProjection : public infinilm::weights::ModelWeights {
public:
    KimiK25VisionPatchProjection(
        infinilm::runtime::Runtime &runtime,
        size_t hidden_size,
        size_t patch_size,
        const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &pixel_values) const;

protected:
    size_t patch_size_{0};
    infinilm::weights::Weight weight_;
    infinilm::weights::Weight bias_;
};

class KimiK25VisionPosEmbed : public infinilm::weights::ModelWeights {
public:
    KimiK25VisionPosEmbed(
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

class KimiK25VisionPatchEmbed : public infinilm::weights::ModelWeights {
public:
    KimiK25VisionPatchEmbed(
        infinilm::runtime::Runtime &runtime,
        const nlohmann::json &config,
        const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &pixel_values,
                                      const infinilm::runtime::Tensor &grid_thw) const;

protected:
    std::shared_ptr<KimiK25VisionPatchProjection> proj_;
    std::shared_ptr<KimiK25VisionPosEmbed> pos_emb_;
};

class KimiK25VisionMLP : public infinilm::weights::ModelWeights {
public:
    KimiK25VisionMLP(
        infinilm::runtime::Runtime &runtime,
        size_t hidden_size,
        size_t intermediate_size,
        const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states) const;

protected:
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> fc0_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> fc1_;
};

class KimiK25VisionBlock : public infinilm::weights::ModelWeights {
public:
    KimiK25VisionBlock(
        infinilm::runtime::Runtime &runtime,
        const nlohmann::json &config,
        std::shared_ptr<infinilm::layers::RoPE> rotary_emb,
        const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      const infinilm::runtime::Tensor &row_positions,
                                      const infinilm::runtime::Tensor &col_positions) const;

protected:
    size_t hidden_size_{0};
    size_t num_heads_{0};
    size_t head_dim_{0};
    float scale_{1.0f};
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb_;

    std::shared_ptr<infinilm::layers::LayerNorm> norm0_;
    std::shared_ptr<infinilm::layers::LayerNorm> norm1_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wqkv_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> wo_;
    std::shared_ptr<KimiK25VisionMLP> mlp_;
};

class KimiK25VisionEncoder : public infinilm::weights::ModelWeights {
public:
    KimiK25VisionEncoder(
        infinilm::runtime::Runtime &runtime,
        const nlohmann::json &config,
        const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      const infinilm::runtime::Tensor &grid_thw) const;

protected:
    std::vector<std::shared_ptr<KimiK25VisionBlock>> blocks_;
    std::shared_ptr<infinilm::layers::LayerNorm> final_layernorm_;
};

class KimiK25VisionTower : public infinilm::weights::ModelWeights {
public:
    KimiK25VisionTower(
        infinilm::runtime::Runtime &runtime,
        const nlohmann::json &config,
        const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &pixel_values,
                                      const infinilm::runtime::Tensor &grid_thw) const;

protected:
    size_t merge_height_{2};
    size_t merge_width_{2};
    size_t hidden_size_{0};
    std::shared_ptr<KimiK25VisionPatchEmbed> patch_embed_;
    std::shared_ptr<KimiK25VisionEncoder> encoder_;
};

class KimiK25Projector : public infinilm::weights::ModelWeights {
public:
    KimiK25Projector(
        infinilm::runtime::Runtime &runtime,
        const nlohmann::json &config,
        const infinilm::runtime::DataType &dtype);
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &vision_features) const;

protected:
    size_t merged_hidden_size_{0};
    std::shared_ptr<infinilm::layers::LayerNorm> pre_norm_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> linear_0_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> linear_2_;
};

} // namespace infinilm::models::kimi_k25
