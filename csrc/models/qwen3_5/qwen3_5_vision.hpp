#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/common_modules.hpp"

#include "../../layers/embedding/embedding.hpp"
#include "../../layers/normalization/layer_norm.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include "runtime/tensor.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_5 {

class Qwen35VisionPatchProj : public infinilm::weights::ModelWeights {
public:
    Qwen35VisionPatchProj(infinilm::runtime::Runtime &runtime, size_t in_channels,
                          size_t hidden_size,
                          size_t temporal_patch_size,
                          size_t patch_size,
                          const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

private:
    size_t in_channels_;
    size_t hidden_size_;
    size_t temporal_patch_size_;
    size_t patch_size_;

    infinilm::weights::Weight weight_;
    infinilm::weights::Weight bias_;
};

class Qwen35VisionPatchEmbed : public infinilm::weights::ModelWeights {
public:
    Qwen35VisionPatchEmbed(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                           const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

private:
    std::shared_ptr<Qwen35VisionPatchProj> proj_;
};

class Qwen35VisionAttention : public infinilm::weights::ModelWeights {
public:
    Qwen35VisionAttention(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                          const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                               const infinilm::runtime::Tensor &row_position_ids,
                               const infinilm::runtime::Tensor &col_position_ids) const;

private:
    size_t hidden_size_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;

    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> qkv_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> proj_;
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb_;
};

class Qwen35VisionMLP : public infinilm::weights::ModelWeights {
public:
    Qwen35VisionMLP(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                    const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

private:
    std::string activation_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> linear_fc1_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> linear_fc2_;
};

class Qwen35VisionBlock : public infinilm::weights::ModelWeights {
public:
    Qwen35VisionBlock(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                      const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                               const infinilm::runtime::Tensor &row_position_ids,
                               const infinilm::runtime::Tensor &col_position_ids) const;

private:
    std::shared_ptr<infinilm::layers::LayerNorm> norm1_;
    std::shared_ptr<infinilm::layers::LayerNorm> norm2_;
    std::shared_ptr<Qwen35VisionAttention> attn_;
    std::shared_ptr<Qwen35VisionMLP> mlp_;
};

class Qwen35VisionPatchMerger : public infinilm::weights::ModelWeights {
public:
    Qwen35VisionPatchMerger(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                            const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const;

private:
    size_t hidden_size_;
    size_t merged_size_;
    std::shared_ptr<infinilm::layers::LayerNorm> norm_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> linear_fc1_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> linear_fc2_;
};

class Qwen35VisionModel : public infinilm::weights::ModelWeights {
public:
    Qwen35VisionModel(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                      const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values,
                               const infinilm::runtime::Tensor &image_grid_thw) const;

private:
    infinilm::runtime::Tensor fast_pos_embed_interpolate(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &image_grid_thw) const;
    infinilm::runtime::Tensor build_rotary_position_ids(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &image_grid_thw) const;

    size_t hidden_size_;
    size_t num_heads_;
    size_t head_dim_;
    size_t spatial_merge_size_;
    size_t num_grid_per_side_;

    std::shared_ptr<Qwen35VisionPatchEmbed> patch_embed_;
    std::shared_ptr<infinilm::layers::Embedding> pos_embed_;
    std::vector<std::shared_ptr<Qwen35VisionBlock>> blocks_;
    std::shared_ptr<Qwen35VisionPatchMerger> merger_;
};

} // namespace infinilm::models::qwen3_5
