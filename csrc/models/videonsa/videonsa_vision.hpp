#pragma once

#include "../../layers/linear/linear.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"
#include "runtime/tensor.hpp"
#include <memory>
#include <nlohmann/json.hpp>
#include <unordered_set>
#include <vector>

namespace infinilm::models::videonsa {

class VideoNSAPatchEmbed : public infinilm::weights::ModelWeights {
public:
    VideoNSAPatchEmbed(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                       const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &pixel_values) const;

private:
    size_t out_hidden_size_;
    size_t patch_dim_;
    infinilm::weights::Weight proj_weight_;
};

class VideoNSAVisionAttention : public infinilm::weights::ModelWeights {
public:
    VideoNSAVisionAttention(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                            const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      const std::vector<int64_t> *cu_window_seqlens = nullptr) const;

private:
    size_t hidden_size_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;

    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> qkv_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> proj_;
};

class VideoNSAVisionMLP : public infinilm::weights::ModelWeights {
public:
    VideoNSAVisionMLP(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                      const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states) const;

private:
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> gate_proj_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> up_proj_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> down_proj_;
};

class VideoNSAVisionBlock : public infinilm::weights::ModelWeights {
public:
    VideoNSAVisionBlock(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                        const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      const std::vector<int64_t> *cu_window_seqlens = nullptr) const;

private:
    std::shared_ptr<infinilm::layers::RMSNorm> norm1_;
    std::shared_ptr<VideoNSAVisionAttention> attn_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm2_;
    std::shared_ptr<VideoNSAVisionMLP> mlp_;
};

class VideoNSAPatchMerger : public infinilm::weights::ModelWeights {
public:
    VideoNSAPatchMerger(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                        const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states) const;

private:
    size_t hidden_size_;
    size_t spatial_merge_unit_;
    size_t merged_size_;

    std::shared_ptr<infinilm::layers::RMSNorm> ln_q_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> mlp_0_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> mlp_2_;
};

class VideoNSAVisionModel : public infinilm::weights::ModelWeights {
public:
    VideoNSAVisionModel(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                        const infinilm::runtime::DataType &dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &pixel_values,
                                      const infinilm::runtime::Tensor &grid_thw) const;

private:
    struct WindowMetadata {
        std::vector<int64_t> patch_order;
        std::vector<int64_t> reverse_order;
        std::vector<int64_t> cu_window_seqlens;
        std::vector<int64_t> cu_full_seqlens;
    };

    WindowMetadata build_window_metadata_(infinilm::runtime::Runtime &runtime,
                                          const infinilm::runtime::Tensor &grid_thw) const;
    infinilm::runtime::Tensor gather_rows_(infinilm::runtime::Runtime &runtime,
                                           const infinilm::runtime::Tensor &hidden_states,
                                           const std::vector<int64_t> &row_order) const;

    size_t depth_;
    size_t hidden_size_;
    size_t patch_size_;
    size_t spatial_merge_size_;
    size_t spatial_merge_unit_;
    size_t window_size_;
    std::unordered_set<size_t> fullatt_block_indexes_;

    std::shared_ptr<VideoNSAPatchEmbed> patch_embed_;
    std::vector<std::shared_ptr<VideoNSAVisionBlock>> blocks_;
    std::shared_ptr<VideoNSAPatchMerger> merger_;
};

} // namespace infinilm::models::videonsa
