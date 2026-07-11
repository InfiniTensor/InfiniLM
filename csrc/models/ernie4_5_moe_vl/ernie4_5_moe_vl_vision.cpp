#include "ernie4_5_moe_vl_vision.hpp"

#include "../../utils.hpp"
#include "infinicore/ops.hpp"

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

// ---------------------------------------------------------------------------
// Patch embedding (visual.patch_embed.*)
// proj is nn::Linear([1280, 588]); input is flattened from [N, C, pH, pW].
// ---------------------------------------------------------------------------
Ernie4_5_VisionPatchEmbed::Ernie4_5_VisionPatchEmbed(const nlohmann::json &vision_config,
                                                     const infinicore::DataType &dtype,
                                                     const infinicore::Device &device) {
    size_t in_channels = vision_config.value("in_channels", 3);
    size_t embed_dim = vision_config.value("embed_dim", 1280);
    size_t patch_size = vision_config.value("patch_size", 14);
    size_t in_features = in_channels * patch_size * patch_size;
    // Checkpoint has no patch_embed.proj.bias -> nn.Linear(bias=False).
    INFINICORE_NN_MODULE_INIT(proj, in_features, embed_dim, false, dtype, device);
}

infinicore::Tensor Ernie4_5_VisionPatchEmbed::forward(const infinicore::Tensor &pixel_values) const {
    // [num_patches, C, pH, pW] -> [num_patches, C*pH*pW] -> [num_patches, embed_dim]
    size_t num_patches = pixel_values->shape()[0];
    size_t flat_size = 1;
    for (size_t i = 1; i < pixel_values->ndim(); ++i) flat_size *= pixel_values->shape()[i];
    auto flat = const_cast<infinicore::Tensor &>(pixel_values)->view({num_patches, flat_size});
    return proj_->forward(flat);
}

// ---------------------------------------------------------------------------
// Attention (2D rope, block-diagonal over images)
// ---------------------------------------------------------------------------
Ernie4_5_VisionAttention::Ernie4_5_VisionAttention(const nlohmann::json &vision_config,
                                                   const infinicore::DataType &dtype,
                                                   const infinicore::Device &device)
    : embed_dim_(vision_config.value("embed_dim", 1280)),
      num_heads_(vision_config.value("num_heads", 16)),
      head_dim_(embed_dim_ / num_heads_),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    INFINICORE_NN_MODULE_INIT(qkv, embed_dim_, embed_dim_ * 3, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(proj, embed_dim_, embed_dim_, true, dtype, device);
}

infinicore::Tensor Ernie4_5_VisionAttention::forward(const infinicore::Tensor &hidden_states,
                                                     const infinicore::Tensor &sin_tbl,
                                                     const infinicore::Tensor &cos_tbl,
                                                     const infinicore::Tensor &pos_index,
                                                     const std::vector<int64_t> &cu_seqlens) const {
    // Input: hidden_states [num_patches, embed_dim] (batchless; cu_seqlens encodes
    // per-frame segmentation). Output: [num_patches, embed_dim].
    ASSERT(hidden_states->ndim() == 2);
    size_t num_patches = hidden_states->shape()[0];

    // Fused QKV projection: out [num_patches, embed_dim * 3].
    auto qkv_out = qkv_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    auto qkv_view = qkv_out->view({num_patches, 3, num_heads_, head_dim_});

    auto q = qkv_view->narrow({{1, 0, 1}})->squeeze(1)->contiguous();  // [N, H, D]
    auto k = qkv_view->narrow({{1, 1, 1}})->squeeze(1)->contiguous();
    auto v = qkv_view->narrow({{1, 2, 1}})->squeeze(1)->contiguous();

    // 2D rope (NEOX rotate_half) over head_dim, per patch position. sin/cos tables
    // already encode (height,width) coords in their two halves.
    infinicore::op::rope_(q, q, pos_index, sin_tbl, cos_tbl, infinicore::nn::RoPE::Algo::GPT_NEOX);
    infinicore::op::rope_(k, k, pos_index, sin_tbl, cos_tbl, infinicore::nn::RoPE::Algo::GPT_NEOX);

    // Block-diagonal attention (attn_sep=true): scaled dot-product runs
    // independently within each frame's patch span [s,e). A single image is one
    // segment, so this reduces to plain full attention.
    auto out = infinicore::Tensor::empty({num_patches, num_heads_, head_dim_}, q->dtype(), q->device());
    for (size_t seg = 0; seg + 1 < cu_seqlens.size(); ++seg) {
        size_t s = static_cast<size_t>(cu_seqlens[seg]);
        size_t e = static_cast<size_t>(cu_seqlens[seg + 1]);
        if (e <= s) {
            continue;
        }
        size_t n = e - s;
        auto qs = q->narrow({{0, s, n}})->permute({1, 0, 2})->contiguous();  // [H, n, D]
        auto ks = k->narrow({{0, s, n}})->permute({1, 0, 2})->contiguous();  // [H, n, D]
        auto vs = v->narrow({{0, s, n}})->permute({1, 0, 2})->contiguous();  // [H, n, D]
        auto ks_t = ks->permute({0, 2, 1});                                  // [H, D, n]

        auto scores = infinicore::op::matmul(qs, ks_t, scale_);              // [H, n, n]
        auto probs = infinicore::op::softmax(scores, -1);                    // [H, n, n]
        auto out_seg = infinicore::op::matmul(probs, vs);                    // [H, n, D]
        out_seg = out_seg->permute({1, 0, 2})->contiguous();                 // [n, H, D]
        out->narrow({{0, s, n}})->copy_from(out_seg);
    }

    auto out_flat = out->view({num_patches, embed_dim_});
    return proj_->forward(out_flat);
}

// ---------------------------------------------------------------------------
// MLP (quick_gelu)
// ---------------------------------------------------------------------------
Ernie4_5_VisionMLP::Ernie4_5_VisionMLP(const nlohmann::json &vision_config,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device) {
    size_t embed_dim = vision_config.value("embed_dim", 1280);
    size_t mlp_ratio = vision_config.value("mlp_ratio", 4);
    size_t intermediate = embed_dim * mlp_ratio;
    INFINICORE_NN_MODULE_INIT(fc1, embed_dim, intermediate, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(fc2, intermediate, embed_dim, true, dtype, device);
}

infinicore::Tensor Ernie4_5_VisionMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto x = fc1_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    // TODO(ernie-vl): quick_gelu(x) = x * sigmoid(1.702 * x). Confirm whether
    // infinicore::op exposes quick_gelu directly; otherwise compose from sigmoid.
    x = infinicore::op::quick_gelu(x);
    return fc2_->forward(x);
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------
Ernie4_5_VisionBlock::Ernie4_5_VisionBlock(const nlohmann::json &vision_config,
                                           const infinicore::DataType &dtype,
                                           const infinicore::Device &device) {
    size_t embed_dim = vision_config.value("embed_dim", 1280);
    float layer_norm_eps = vision_config.value("layer_norm_eps", 1e-6f);
    INFINICORE_NN_MODULE_INIT(norm1, embed_dim, layer_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, vision_config, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm2, embed_dim, layer_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, vision_config, dtype, device);
}

infinicore::Tensor Ernie4_5_VisionBlock::forward(const infinicore::Tensor &hidden_states,
                                                 const infinicore::Tensor &sin_tbl,
                                                 const infinicore::Tensor &cos_tbl,
                                                 const infinicore::Tensor &pos_index,
                                                 const std::vector<int64_t> &cu_seqlens) const {
    auto h = const_cast<infinicore::Tensor &>(hidden_states);
    auto normed = norm1_->forward(h);
    auto attn_out = attn_->forward(normed, sin_tbl, cos_tbl, pos_index, cu_seqlens);
    h = infinicore::op::add(h, attn_out);

    auto normed2 = norm2_->forward(h);
    auto mlp_out = mlp_->forward(normed2);
    return infinicore::op::add(h, mlp_out);
}

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------
Ernie4_5_VisionTransformer::Ernie4_5_VisionTransformer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    const nlohmann::json &vision_config = model_config->get_config_json().at("vision_config");
    const auto &dtype = model_config->get_dtype();

    embed_dim_ = vision_config.value("embed_dim", 1280);
    num_heads_ = vision_config.value("num_heads", 16);
    head_dim_ = embed_dim_ / num_heads_;
    spatial_merge_size_ = vision_config.value("spatial_merge_size", 2);
    // VERIFY(GPU): vision 2D-rope base frequency; Qwen2-VL uses 10000. Confirm
    // against HF DFNRope config (key may differ).
    rope_theta_vision_ = vision_config.value("rope_theta", 10000.0);

    INFINICORE_NN_MODULE_INIT(patch_embed, vision_config, dtype, device);

    size_t depth = vision_config.value("depth", 32);
    blocks_.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        blocks_.push_back(this->register_module<Ernie4_5_VisionBlock>(
            "blocks." + std::to_string(i), vision_config, dtype, device));
    }

    // Post-transformer LayerNorm (visual.norm1.*) applied before the merger.
    float layer_norm_eps = vision_config.value("layer_norm_eps", 1e-6f);
    INFINICORE_NN_MODULE_INIT(norm1, embed_dim_, layer_norm_eps, dtype, device);

    // Adapter: registered as "merger" to match HF checkpoint prefix visual.merger.*
    INFINICORE_NN_MODULE_INIT(merger, model_config, device);
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
Ernie4_5_VisionTransformer::build_rope_(const infinicore::Tensor &grid_thw,
                                        const infinicore::DataType &dtype,
                                        const infinicore::Device &device) const {
    // grid_thw: [num_img, 3] = (t, h, w) in patch units (int64). For each patch
    // compute its (h_coord, w_coord) in the merge-block order the processor emits:
    // for each (hb, wb) block, the m*m patches in (mh, mw) order; coord = block*m+sub.
    auto thw = grid_thw->to(infinicore::Device::cpu())->contiguous();
    const auto *g = reinterpret_cast<const int64_t *>(thw->data());
    size_t num_img = thw->shape()[0];
    int64_t m = static_cast<int64_t>(spatial_merge_size_);

    std::vector<int64_t> hpos;
    std::vector<int64_t> wpos;
    for (size_t im = 0; im < num_img; ++im) {
        int64_t t = g[im * 3 + 0];
        int64_t h = g[im * 3 + 1];
        int64_t w = g[im * 3 + 2];
        for (int64_t frame = 0; frame < t; ++frame) {
            for (int64_t hb = 0; hb < h / m; ++hb) {
                for (int64_t wb = 0; wb < w / m; ++wb) {
                    for (int64_t mh = 0; mh < m; ++mh) {
                        for (int64_t mw = 0; mw < m; ++mw) {
                            hpos.push_back(hb * m + mh);
                            wpos.push_back(wb * m + mw);
                        }
                    }
                }
            }
        }
    }
    size_t num_patches = hpos.size();

    size_t cache_dim = head_dim_ / 2; // 40 for head_dim 80
    size_t half = cache_dim / 2;      // 20 freqs per axis (height / width)
    std::vector<float> sin_f(num_patches * cache_dim);
    std::vector<float> cos_f(num_patches * cache_dim);
    for (size_t k = 0; k < half; ++k) {
        // VisionRotaryEmbedding inv_freq over dim=cache_dim: theta^(-2k/cache_dim).
        float inv_freq = 1.0f / std::pow(static_cast<float>(rope_theta_vision_),
                                         2.0f * static_cast<float>(k) / static_cast<float>(cache_dim));
        for (size_t i = 0; i < num_patches; ++i) {
            float ah = static_cast<float>(hpos[i]) * inv_freq; // first half: height
            float aw = static_cast<float>(wpos[i]) * inv_freq; // second half: width
            sin_f[i * cache_dim + k] = std::sin(ah);
            cos_f[i * cache_dim + k] = std::cos(ah);
            sin_f[i * cache_dim + half + k] = std::sin(aw);
            cos_f[i * cache_dim + half + k] = std::cos(aw);
        }
    }

    auto to_table = [&](const std::vector<float> &f) -> infinicore::Tensor {
        auto out = infinicore::Tensor::empty({num_patches, cache_dim}, dtype, device);
        if (dtype == infinicore::DataType::F32) {
            auto cpu = infinicore::Tensor::from_blob(const_cast<float *>(f.data()), {num_patches, cache_dim},
                                                     infinicore::DataType::F32, infinicore::Device::cpu());
            out->copy_from(cpu);
        } else if (dtype == infinicore::DataType::BF16) {
            std::vector<uint16_t> hbuf(f.size());
            for (size_t i = 0; i < f.size(); ++i) {
                hbuf[i] = f32_to_bf16(f[i]);
            }
            auto cpu = infinicore::Tensor::from_blob(hbuf.data(), {num_patches, cache_dim},
                                                     infinicore::DataType::BF16, infinicore::Device::cpu());
            out->copy_from(cpu);
        } else if (dtype == infinicore::DataType::F16) {
            std::vector<uint16_t> hbuf(f.size());
            for (size_t i = 0; i < f.size(); ++i) {
                hbuf[i] = f32_to_f16(f[i]);
            }
            auto cpu = infinicore::Tensor::from_blob(hbuf.data(), {num_patches, cache_dim},
                                                     infinicore::DataType::F16, infinicore::Device::cpu());
            out->copy_from(cpu);
        } else {
            throw std::runtime_error("build_rope_: unsupported dtype for vision rope tables");
        }
        return out;
    };

    auto sin_tbl = to_table(sin_f);
    auto cos_tbl = to_table(cos_f);

    std::vector<int64_t> idx(num_patches);
    for (size_t i = 0; i < num_patches; ++i) {
        idx[i] = static_cast<int64_t>(i);
    }
    auto idx_cpu = infinicore::Tensor::from_blob(idx.data(), {num_patches}, infinicore::DataType::I64,
                                                 infinicore::Device::cpu());
    auto pos_index = idx_cpu->to(device);

    return std::make_tuple(sin_tbl, cos_tbl, pos_index);
}

std::vector<int64_t>
Ernie4_5_VisionTransformer::build_cu_seqlens_(const infinicore::Tensor &grid_thw) const {
    // Per-frame attention segments (Qwen2-VL style): each frame's h*w patches form
    // one block-diagonal segment. Boundaries are a cumulative sum [0, ...].
    auto thw = grid_thw->to(infinicore::Device::cpu())->contiguous();
    const auto *g = reinterpret_cast<const int64_t *>(thw->data());
    size_t num_img = thw->shape()[0];

    std::vector<int64_t> cu;
    cu.push_back(0);
    int64_t acc = 0;
    for (size_t im = 0; im < num_img; ++im) {
        int64_t t = g[im * 3 + 0];
        int64_t h = g[im * 3 + 1];
        int64_t w = g[im * 3 + 2];
        for (int64_t frame = 0; frame < t; ++frame) {
            acc += h * w;
            cu.push_back(acc);
        }
    }
    return cu;
}

infinicore::Tensor Ernie4_5_VisionTransformer::forward(const infinicore::Tensor &pixel_values,
                                                       const infinicore::Tensor &grid_thw) const {
    // pixel_values: [num_patches, in_channels, patch, patch] (NCHW per-patch view)
    //   produced by the processor's patchify step.
    // grid_thw: [num_images, 3] = (t, h_in_patches, w_in_patches), CPU side.
    //
    // Linear patch embedding: [num_patches, C*pH*pW] -> [num_patches, embed_dim].
    auto patch_out = patch_embed_->forward(pixel_values);
    ASSERT(patch_out->ndim() == 2);
    auto hidden = patch_out;

    // 2. 2D-rope tables + per-frame attention segments from the patch grid.
    auto [sin_tbl, cos_tbl, pos_index] = build_rope_(grid_thw, hidden->dtype(), hidden->device());
    auto cu_seqlens = build_cu_seqlens_(grid_thw);

    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        hidden = blocks_[bi]->forward(hidden, sin_tbl, cos_tbl, pos_index, cu_seqlens);
    }

    // 3. Post-transformer LayerNorm (visual.norm1).
    hidden = norm1_->forward(hidden);

    // 4. Spatial+temporal merge and projection -> [num_merged_tokens, text_hidden_size].
    return merger_->forward(hidden, grid_thw);
}

} // namespace infinilm::models::ernie4_5_moe_vl
