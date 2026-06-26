#pragma once

#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <nlohmann/json.hpp>
#include <unordered_set>
#include <vector>

namespace infinilm::models::videonsa {

class VideoNSAPatchEmbed : public infinicore::nn::Module {
public:
    VideoNSAPatchEmbed(const nlohmann::json &config,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values) const;

private:
    size_t out_hidden_size_;
    size_t patch_dim_;
    INFINICORE_NN_PARAMETER(proj_weight);
};

class VideoNSAVisionAttention : public infinicore::nn::Module {
public:
    VideoNSAVisionAttention(const nlohmann::json &config,
                            const infinicore::DataType &dtype,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const std::vector<int64_t> *cu_window_seqlens = nullptr) const;

private:
    size_t hidden_size_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;

    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, qkv);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, proj);
};

class VideoNSAVisionMLP : public infinicore::nn::Module {
public:
    VideoNSAVisionMLP(const nlohmann::json &config,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, gate_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, up_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, down_proj);
};

class VideoNSAVisionBlock : public infinicore::nn::Module {
public:
    VideoNSAVisionBlock(const nlohmann::json &config,
                        const infinicore::DataType &dtype,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const std::vector<int64_t> *cu_window_seqlens = nullptr) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm1);
    INFINICORE_NN_MODULE(VideoNSAVisionAttention, attn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm2);
    INFINICORE_NN_MODULE(VideoNSAVisionMLP, mlp);
};

class VideoNSAPatchMerger : public infinicore::nn::Module {
public:
    VideoNSAPatchMerger(const nlohmann::json &config,
                        const infinicore::DataType &dtype,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    size_t hidden_size_;
    size_t spatial_merge_unit_;
    size_t merged_size_;

    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, ln_q);
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> mlp_0_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> mlp_2_;
};

class VideoNSAVisionModel : public infinicore::nn::Module {
public:
    VideoNSAVisionModel(const nlohmann::json &config,
                        const infinicore::DataType &dtype,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const infinicore::Tensor &grid_thw) const;

private:
    struct WindowMetadata {
        std::vector<int64_t> patch_order;
        std::vector<int64_t> reverse_order;
        std::vector<int64_t> cu_window_seqlens;
        std::vector<int64_t> cu_full_seqlens;
    };

    WindowMetadata build_window_metadata_(const infinicore::Tensor &grid_thw) const;
    infinicore::Tensor gather_rows_(const infinicore::Tensor &hidden_states,
                                    const std::vector<int64_t> &row_order) const;

    size_t depth_;
    size_t hidden_size_;
    size_t patch_size_;
    size_t spatial_merge_size_;
    size_t spatial_merge_unit_;
    size_t window_size_;
    std::unordered_set<size_t> fullatt_block_indexes_;

    INFINICORE_NN_MODULE(VideoNSAPatchEmbed, patch_embed);
    INFINICORE_NN_MODULE_VEC(VideoNSAVisionBlock, blocks);
    INFINICORE_NN_MODULE(VideoNSAPatchMerger, merger);
};

} // namespace infinilm::models::videonsa
