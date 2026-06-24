#include "qwen3_vl_vision_layers.hpp"

#include "../../config/model_config.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mha.hpp"
#include "infinicore/ops/rope.hpp"

#include <cmath>
#include <optional>

namespace infinilm::models::qwen3_vl {

using infinilm::config::json_size;

Qwen3VLPatchProjection::Qwen3VLPatchProjection(size_t out_features,
                                               size_t in_channels,
                                               size_t temporal_patch_size,
                                               size_t patch_size,
                                               const infinicore::DataType &dtype,
                                               const infinicore::Device &device)
    : patch_dim_(in_channels * temporal_patch_size * patch_size * patch_size) {
    INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_channels, temporal_patch_size, patch_size, patch_size}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype, device));
}

infinicore::Tensor Qwen3VLPatchProjection::forward(const infinicore::Tensor &pixel_values) const {
    auto input = const_cast<infinicore::Tensor &>(pixel_values);
    auto weight_2d = static_cast<const infinicore::Tensor &>(weight_)->view({weight_->size(0), patch_dim_});
    auto bias_tensor = static_cast<const infinicore::Tensor &>(bias_);
    return infinicore::op::linear(input, weight_2d, bias_tensor);
}

Qwen3VLPatchEmbed::Qwen3VLPatchEmbed(const nlohmann::json &config,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(proj,
                              json_size(config, "hidden_size"),
                              json_size(config, "in_channels", 3),
                              json_size(config, "temporal_patch_size", 2),
                              json_size(config, "patch_size", 16),
                              dtype,
                              device);
}

infinicore::Tensor Qwen3VLPatchEmbed::forward(const infinicore::Tensor &pixel_values) const {
    return proj_->forward(pixel_values);
}

Qwen3VLVisionMLP::Qwen3VLVisionMLP(const nlohmann::json &config,
                                   const infinicore::DataType &dtype,
                                   const infinicore::Device &device) {
    size_t hidden_size = json_size(config, "hidden_size");
    size_t intermediate_size = json_size(config, "intermediate_size");
    INFINICORE_NN_MODULE_INIT(linear_fc1, hidden_size, intermediate_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear_fc2, intermediate_size, hidden_size, true, dtype, device);
}

infinicore::Tensor Qwen3VLVisionMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto x = linear_fc1_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    x = infinicore::op::gelu_tanh(x);
    return linear_fc2_->forward(x);
}

Qwen3VLVisionAttention::Qwen3VLVisionAttention(const nlohmann::json &config,
                                               const infinicore::DataType &dtype,
                                               const infinicore::Device &device)
    : hidden_size_(json_size(config, "hidden_size")),
      num_heads_(json_size(config, "num_heads")),
      head_dim_(hidden_size_ / num_heads_),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    INFINICORE_NN_MODULE_INIT(qkv, hidden_size_, hidden_size_ * 3, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(proj, hidden_size_, hidden_size_, true, dtype, device);
}

infinicore::Tensor Qwen3VLVisionAttention::forward(const infinicore::Tensor &hidden_states,
                                                   const infinicore::Tensor &position_ids,
                                                   const infinicore::Tensor &sin_table,
                                                   const infinicore::Tensor &cos_table) const {
    size_t seq_len = hidden_states->size(0);
    auto qkv = qkv_->forward(const_cast<infinicore::Tensor &>(hidden_states))
                   ->view({seq_len, 3, num_heads_, head_dim_});
    auto q = qkv->narrow({{1, 0, 1}})->squeeze(1);
    auto k = qkv->narrow({{1, 1, 1}})->squeeze(1);
    auto v = qkv->narrow({{1, 2, 1}})->squeeze(1)->view({1, seq_len, num_heads_, head_dim_});

    q = infinicore::op::rope(q, position_ids, sin_table, cos_table, infinicore::nn::RoPE::Algo::GPT_NEOX)
            ->view({1, seq_len, num_heads_, head_dim_});
    k = infinicore::op::rope(k, position_ids, sin_table, cos_table, infinicore::nn::RoPE::Algo::GPT_NEOX)
            ->view({1, seq_len, num_heads_, head_dim_});

    auto attn_output = infinicore::op::mha(q, k, v, std::nullopt, scale_, false)
                           ->view({seq_len, hidden_size_});
    return proj_->forward(attn_output);
}

Qwen3VLVisionBlock::Qwen3VLVisionBlock(const nlohmann::json &config,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device) {
    size_t hidden_size = json_size(config, "hidden_size");
    INFINICORE_NN_MODULE_INIT(norm1, hidden_size, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm2, hidden_size, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, config, dtype, device);
}

infinicore::Tensor Qwen3VLVisionBlock::forward(const infinicore::Tensor &hidden_states,
                                               const infinicore::Tensor &position_ids,
                                               const infinicore::Tensor &sin_table,
                                               const infinicore::Tensor &cos_table) const {
    auto residual = hidden_states;
    auto x = norm1_->forward(hidden_states);
    x = attn_->forward(x, position_ids, sin_table, cos_table);
    x = infinicore::op::add(x, residual);

    residual = x;
    x = norm2_->forward(x);
    x = mlp_->forward(x);
    return infinicore::op::add(x, residual);
}

Qwen3VLPatchMerger::Qwen3VLPatchMerger(const nlohmann::json &config,
                                       bool use_postshuffle_norm,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device)
    : hidden_size_(json_size(config, "hidden_size")),
      merged_size_(hidden_size_ * json_size(config, "spatial_merge_size", 2) * json_size(config, "spatial_merge_size", 2)),
      use_postshuffle_norm_(use_postshuffle_norm) {
    size_t out_hidden_size = json_size(config, "out_hidden_size");
    INFINICORE_NN_MODULE_INIT(norm, use_postshuffle_norm_ ? merged_size_ : hidden_size_, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear_fc1, merged_size_, merged_size_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear_fc2, merged_size_, out_hidden_size, true, dtype, device);
}

infinicore::Tensor Qwen3VLPatchMerger::forward(const infinicore::Tensor &hidden_states) const {
    infinicore::Tensor x;
    if (use_postshuffle_norm_) {
        x = hidden_states->view({hidden_states->size(0) / 4, merged_size_});
        x = norm_->forward(x);
    } else {
        x = norm_->forward(hidden_states);
        x = x->view({x->size(0) / 4, merged_size_});
    }
    x = linear_fc1_->forward(x);
    x = infinicore::op::gelu(x);
    return linear_fc2_->forward(x);
}

} // namespace infinilm::models::qwen3_vl
