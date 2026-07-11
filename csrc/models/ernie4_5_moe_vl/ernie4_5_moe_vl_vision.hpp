#pragma once

#include "ernie4_5_moe_vl_resampler.hpp"

#include "../../config/model_config.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <nlohmann/json.hpp>

#include <optional>
#include <tuple>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

// Vision tower: DFNRopeVisionTransformer.
// config.vision_config: depth 32, embed_dim 1280, num_heads 16 (head_dim 80),
// patch_size 14, spatial_merge_size 2, mlp_ratio 4, hidden_act "quick_gelu",
// attn_sep true. Uses 2D rotary position embedding (NaViT-style, variable
// resolution), not learned positional embeddings.

// Linear patch embedding: flatten [num_patches, C, pH, pW] -> [num_patches, C*pH*pW],
// then project via nn::Linear. Weight shape [embed_dim, in_channels*patch_size^2] = [1280, 588].
// "proj" submodule matches checkpoint path visual.patch_embed.proj.{weight,bias}.
class Ernie4_5_VisionPatchEmbed : public infinicore::nn::Module {
public:
    Ernie4_5_VisionPatchEmbed(const nlohmann::json &vision_config,
                              const infinicore::DataType &dtype,
                              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &pixel_values) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Linear, proj);
};

// Self-attention with 2D RoPE. qkv is a single fused projection.
class Ernie4_5_VisionAttention : public infinicore::nn::Module {
public:
    Ernie4_5_VisionAttention(const nlohmann::json &vision_config,
                             const infinicore::DataType &dtype,
                             const infinicore::Device &device);

    // sin_tbl/cos_tbl: precomputed 2D-rope tables [num_patches, head_dim/2].
    // pos_index: [num_patches] arange selecting the table row per patch.
    // cu_seqlens: per-image patch boundaries [num_img+1] for block-diagonal
    // attention (attn_sep) — attention runs independently within each segment.
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &sin_tbl,
                               const infinicore::Tensor &cos_tbl,
                               const infinicore::Tensor &pos_index,
                               const std::vector<int64_t> &cu_seqlens) const;

private:
    size_t embed_dim_;
    size_t num_heads_;
    size_t head_dim_;
    float scale_;

    INFINICORE_NN_MODULE(infinicore::nn::Linear, qkv);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, proj);
};

// MLP with quick_gelu: fc2(quick_gelu(fc1(x))), quick_gelu(x)=x*sigmoid(1.702*x).
class Ernie4_5_VisionMLP : public infinicore::nn::Module {
public:
    Ernie4_5_VisionMLP(const nlohmann::json &vision_config,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Linear, fc1);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, fc2);
};

// One ViT block: norm1 -> attn -> residual -> norm2 -> mlp -> residual.
class Ernie4_5_VisionBlock : public infinicore::nn::Module {
public:
    Ernie4_5_VisionBlock(const nlohmann::json &vision_config,
                         const infinicore::DataType &dtype,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &sin_tbl,
                               const infinicore::Tensor &cos_tbl,
                               const infinicore::Tensor &pos_index,
                               const std::vector<int64_t> &cu_seqlens) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm1);
    INFINICORE_NN_MODULE(Ernie4_5_VisionAttention, attn);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm2);
    INFINICORE_NN_MODULE(Ernie4_5_VisionMLP, mlp);
};

// Full vision transformer + adapter. Registered as "visual" in the top-level
// model to match the HF checkpoint prefix. Includes:
//   - patch embedding (visual.patch_embed.*)
//   - 32 ViT blocks (visual.blocks.*)
//   - VariableResolutionResampler adapter (visual.merger.*)
// Output: merged tokens [num_merged, text_hidden_size] ready for cross-modal fusion.
class Ernie4_5_VisionTransformer : public infinicore::nn::Module {
public:
    // Takes full model_config so both vision_config and text hidden_size
    // (needed by the merger projection) are accessible.
    Ernie4_5_VisionTransformer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               const infinicore::Device &device);

    // pixel_values: flattened patches [num_patches, C, p, p];
    // grid_thw: [num_media, 3] = (t, h, w) in patch units.
    // Returns [num_merged_tokens, text_hidden_size].
    infinicore::Tensor forward(const infinicore::Tensor &pixel_values,
                               const infinicore::Tensor &grid_thw) const;

private:
    // Build DFNRope 2D-rope tables [num_patches, head_dim/2] (height freqs ++ width
    // freqs) and a [num_patches] arange index. Each patch's (h,w) grid coordinate
    // drives the two halves; reused via op::rope (NEOX rotate_half).
    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
    build_rope_(const infinicore::Tensor &grid_thw,
                const infinicore::DataType &dtype,
                const infinicore::Device &device) const;

    // Per-image patch boundaries [num_img+1] (host) for block-diagonal attention.
    std::vector<int64_t> build_cu_seqlens_(const infinicore::Tensor &grid_thw) const;

    size_t embed_dim_;
    size_t num_heads_;
    size_t head_dim_;
    size_t spatial_merge_size_;
    double rope_theta_vision_{10000.0};

    INFINICORE_NN_MODULE(Ernie4_5_VisionPatchEmbed, patch_embed);
    INFINICORE_NN_MODULE_VEC(Ernie4_5_VisionBlock, blocks);
    // Post-transformer LayerNorm: visual.norm1.weight / visual.norm1.bias
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm1);
    // Registered as "merger" to match HF: visual.merger.*
    INFINICORE_NN_MODULE(Ernie4_5_VLResampler, merger);
};

} // namespace infinilm::models::ernie4_5_moe_vl
