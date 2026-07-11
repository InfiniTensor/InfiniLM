#include "ernie4_5_vision.hpp"

#include "infinicore/ops.hpp"
#include "infinicore/ops/mha_varlen.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {
namespace {

struct GridTHW {
    int64_t t;
    int64_t h;
    int64_t w;
};

std::vector<GridTHW> read_grid_thw_cpu(const infinicore::Tensor &grid_thw) {
    auto grid_cpu = grid_thw->to(infinicore::Device::cpu());
    auto shape = grid_cpu->shape();
    if (shape.size() != 2 || shape[1] != 3) {
        throw std::runtime_error("Ernie4_5 VL: grid_thw must have shape [num_images, 3]");
    }

    std::vector<GridTHW> grids;
    grids.reserve(shape[0]);
    for (size_t i = 0; i < shape[0]; ++i) {
        GridTHW grid{0, 0, 0};
        if (grid_cpu->dtype() == infinicore::DataType::I64) {
            const auto *grid_ptr = reinterpret_cast<const int64_t *>(grid_cpu->data());
            grid = {grid_ptr[i * 3 + 0], grid_ptr[i * 3 + 1], grid_ptr[i * 3 + 2]};
        } else if (grid_cpu->dtype() == infinicore::DataType::I32) {
            const auto *grid_ptr = reinterpret_cast<const int32_t *>(grid_cpu->data());
            grid = {grid_ptr[i * 3 + 0], grid_ptr[i * 3 + 1], grid_ptr[i * 3 + 2]};
        } else {
            throw std::runtime_error("Ernie4_5 VL: grid_thw must be int32 or int64");
        }
        if (grid.t <= 0 || grid.h <= 0 || grid.w <= 0) {
            throw std::runtime_error("Ernie4_5 VL: invalid grid_thw");
        }
        grids.push_back(grid);
    }
    return grids;
}

std::pair<infinicore::Tensor, int> build_vision_cu_seqlens(const infinicore::Tensor &grid_thw,
                                                           size_t seq_len) {
    std::vector<int32_t> cu_seqlens;
    cu_seqlens.push_back(0);
    int max_seqlen = 0;
    size_t total = 0;
    for (const auto &grid : read_grid_thw_cpu(grid_thw)) {
        const int segment_len = static_cast<int>(grid.h * grid.w);
        max_seqlen = std::max(max_seqlen, segment_len);
        for (int64_t ti = 0; ti < grid.t; ++ti) {
            (void)ti;
            total += static_cast<size_t>(segment_len);
            cu_seqlens.push_back(static_cast<int32_t>(total));
        }
    }
    if (total != seq_len) {
        throw std::runtime_error("Ernie4_5VisionAttention: grid_thw does not match sequence length");
    }
    auto cu_cpu = infinicore::Tensor::from_blob(
        cu_seqlens.data(),
        {cu_seqlens.size()},
        infinicore::DataType::I32,
        infinicore::Device::cpu());
    return {cu_cpu->to(grid_thw->device()), max_seqlen};
}

} // namespace

Ernie4_5VisionPatchEmbed::Ernie4_5VisionPatchEmbed(const nlohmann::json &vision_config,
                                                   const infinicore::DataType &dtype,
                                                   const infinicore::Device &device) {
    const size_t in_chans = vision_config.value("in_chans", vision_config.value("in_channels", 3));
    const size_t patch_size = vision_config.value("patch_size", vision_config.value("spatial_patch_size", 14));
    const size_t embed_dim = vision_config.value("embed_dim", vision_config.value("hidden_size", 1280));
    INFINICORE_NN_MODULE_INIT(proj, in_chans * patch_size * patch_size, embed_dim, false, dtype, device);
}

infinicore::Tensor Ernie4_5VisionPatchEmbed::forward(const infinicore::Tensor &pixel_values) const {
    return proj_->forward(const_cast<infinicore::Tensor &>(pixel_values));
}

Ernie4_5VisionMLP::Ernie4_5VisionMLP(size_t hidden_size,
                                     double mlp_ratio,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device) {
    const size_t intermediate_size = static_cast<size_t>(static_cast<double>(hidden_size) * mlp_ratio);
    INFINICORE_NN_MODULE_INIT(fc1, hidden_size, intermediate_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(fc2, intermediate_size, hidden_size, true, dtype, device);
}

infinicore::Tensor Ernie4_5VisionMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto x = fc1_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    x = infinicore::op::quick_gelu(x);
    return fc2_->forward(x);
}

Ernie4_5VisionAttention::Ernie4_5VisionAttention(size_t hidden_size,
                                                 size_t num_heads,
                                                 const infinicore::DataType &dtype,
                                                 const infinicore::Device &device)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      head_dim_(hidden_size / num_heads),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    if (hidden_size_ % num_heads_ != 0) {
        throw std::runtime_error("Ernie4_5VisionAttention: hidden_size must be divisible by num_heads");
    }
    INFINICORE_NN_MODULE_INIT(qkv, hidden_size_, 3 * hidden_size_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(proj, hidden_size_, hidden_size_, true, dtype, device);
}

infinicore::Tensor Ernie4_5VisionAttention::forward(const infinicore::Tensor &hidden_states,
                                                    const infinicore::Tensor &rotary_pos_ids,
                                                    const infinicore::Tensor &grid_thw) const {
    const size_t seq_len = hidden_states->size(0);
    auto qkv = qkv_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    auto q = qkv->narrow({{1, 0, hidden_size_}})
                 ->view({seq_len, num_heads_, head_dim_})
                 ->contiguous();
    auto k = qkv->narrow({{1, hidden_size_, hidden_size_}})
                 ->view({seq_len, num_heads_, head_dim_})
                 ->contiguous();
    auto v = qkv->narrow({{1, 2 * hidden_size_, hidden_size_}})
                 ->view({seq_len, num_heads_, head_dim_})
                 ->contiguous();

    infinicore::op::ernie45_vision_rope_(q, k, rotary_pos_ids, 10000.0);
    auto [cu_seqlens, max_seqlen] = build_vision_cu_seqlens(grid_thw, seq_len);
    auto out_tokens = infinicore::op::mha_varlen(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        std::nullopt,
        max_seqlen,
        max_seqlen,
        std::nullopt,
        scale_);
    auto out = out_tokens->view({seq_len, hidden_size_})->contiguous();
    return proj_->forward(out);
}

Ernie4_5VisionBlock::Ernie4_5VisionBlock(const nlohmann::json &vision_config,
                                         const infinicore::DataType &dtype,
                                         const infinicore::Device &device) {
    const size_t hidden_size = vision_config.value("hidden_size", vision_config.value("embed_dim", 1280));
    const size_t num_heads = vision_config.value("num_heads", 16);
    const double mlp_ratio = vision_config.value("mlp_ratio", 4.0);
    INFINICORE_NN_MODULE_INIT(norm1, hidden_size, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, hidden_size, num_heads, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm2, hidden_size, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, hidden_size, mlp_ratio, dtype, device);
}

infinicore::Tensor Ernie4_5VisionBlock::forward(const infinicore::Tensor &hidden_states,
                                                const infinicore::Tensor &rotary_pos_ids,
                                                const infinicore::Tensor &grid_thw) const {
    auto residual = hidden_states;
    auto x = norm1_->forward(hidden_states);
    x = attn_->forward(x, rotary_pos_ids, grid_thw);
    x = infinicore::op::add(x, residual);

    residual = x;
    x = norm2_->forward(x);
    x = mlp_->forward(x);
    return infinicore::op::add(x, residual);
}

Ernie4_5VisionTransformer::Ernie4_5VisionTransformer(const nlohmann::json &vision_config,
                                                     double norm_eps,
                                                     const infinicore::DataType &dtype,
                                                     const infinicore::Device &device)
    : spatial_merge_size_(vision_config.value("spatial_merge_size", 2)) {
    const size_t hidden_size = vision_config.value("hidden_size", vision_config.value("embed_dim", 1280));
    const size_t depth = vision_config.value("depth", 32);
    INFINICORE_NN_MODULE_INIT(patch_embed, vision_config, dtype, device);
    INFINICORE_NN_MODULE_VEC_INIT(blocks, depth, Ernie4_5VisionBlock, vision_config, dtype, device);
    INFINICORE_NN_MODULE_INIT(ln, hidden_size, norm_eps, dtype, device);
}

infinicore::Tensor Ernie4_5VisionTransformer::forward(const infinicore::Tensor &pixel_values,
                                                      const infinicore::Tensor &grid_thw) const {
    auto hidden_states = patch_embed_->forward(pixel_values);
    auto rotary_pos_ids = build_rotary_pos_ids(grid_thw, hidden_states->size(0));
    for (const auto &block : blocks_) {
        hidden_states = block->forward(hidden_states, rotary_pos_ids, grid_thw);
    }
    return ln_->forward(hidden_states);
}

infinicore::Tensor Ernie4_5VisionTransformer::build_rotary_pos_ids(const infinicore::Tensor &grid_thw,
                                                                   size_t seq_len) const {
    std::vector<int32_t> pos;
    pos.reserve(seq_len * 2);
    for (const auto &grid : read_grid_thw_cpu(grid_thw)) {
        const int64_t t = grid.t;
        const int64_t h = grid.h;
        const int64_t w = grid.w;
        if (t <= 0 || h <= 0 || w <= 0 || h % static_cast<int64_t>(spatial_merge_size_) != 0 || w % static_cast<int64_t>(spatial_merge_size_) != 0) {
            throw std::runtime_error("Ernie4_5VisionTransformer: invalid grid_thw");
        }
        for (int64_t ti = 0; ti < t; ++ti) {
            (void)ti;
            for (int64_t hb = 0; hb < h; hb += static_cast<int64_t>(spatial_merge_size_)) {
                for (int64_t wb = 0; wb < w; wb += static_cast<int64_t>(spatial_merge_size_)) {
                    for (size_t hs = 0; hs < spatial_merge_size_; ++hs) {
                        for (size_t ws = 0; ws < spatial_merge_size_; ++ws) {
                            pos.push_back(static_cast<int32_t>(hb + static_cast<int64_t>(hs)));
                            pos.push_back(static_cast<int32_t>(wb + static_cast<int64_t>(ws)));
                        }
                    }
                }
            }
        }
    }
    if (pos.size() != seq_len * 2) {
        throw std::runtime_error("Ernie4_5VisionTransformer: rotary position count does not match vision sequence length");
    }
    auto pos_cpu = infinicore::Tensor::from_blob(
        pos.data(), {seq_len, 2}, infinicore::DataType::I32, infinicore::Device::cpu());
    return pos_cpu->to(grid_thw->device());
}

Ernie4_5VariableResolutionResampler::Ernie4_5VariableResolutionResampler(
    const nlohmann::json &config,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : in_dim_(config.value("pixel_hidden_size", 1280)),
      out_dim_(config.value("hidden_size", 2560)),
      spatial_conv_size_(config.value("spatial_conv_size", 2)),
      temporal_conv_size_(config.value("temporal_conv_size", 2)),
      use_temporal_conv_(config.value("use_temporal_conv", true)) {
    const size_t spatial_dim = in_dim_ * spatial_conv_size_ * spatial_conv_size_;
    const size_t temporal_dim = spatial_dim * temporal_conv_size_;
    spatial_linear0_ = this->register_module<infinicore::nn::Linear>("spatial_linear.0", spatial_dim, spatial_dim, true, dtype, device);
    spatial_linear2_ = this->register_module<infinicore::nn::Linear>("spatial_linear.2", spatial_dim, spatial_dim, true, dtype, device);
    spatial_linear3_ = this->register_module<infinicore::nn::LayerNorm>("spatial_linear.3", spatial_dim, 1e-6, dtype, device);
    if (use_temporal_conv_) {
        temporal_linear0_ = this->register_module<infinicore::nn::Linear>("temporal_linear.0", temporal_dim, spatial_dim, true, dtype, device);
        temporal_linear2_ = this->register_module<infinicore::nn::Linear>("temporal_linear.2", spatial_dim, spatial_dim, true, dtype, device);
        temporal_linear3_ = this->register_module<infinicore::nn::LayerNorm>("temporal_linear.3", spatial_dim, 1e-6, dtype, device);
    }
    mlp_ = this->register_module<infinicore::nn::Linear>("mlp", spatial_dim, out_dim_, true, dtype, device);
    after_norm_ = this->register_module<infinicore::nn::RMSNorm>("after_norm", out_dim_, config.value("rms_norm_eps", 1e-6), dtype, device);
}

infinicore::Tensor Ernie4_5VariableResolutionResampler::spatial_forward(const infinicore::Tensor &hidden_states) const {
    const size_t seq_len = hidden_states->size(0);
    const size_t pack = spatial_conv_size_ * spatial_conv_size_;
    if (seq_len % pack != 0) {
        throw std::runtime_error("Ernie4_5VariableResolutionResampler: vision seq_len is not divisible by spatial_conv_size^2");
    }
    auto x = hidden_states->contiguous()->view({seq_len / pack, in_dim_ * pack});
    x = spatial_linear0_->forward(x);
    x = infinicore::op::gelu(x);
    x = spatial_linear2_->forward(x);
    return spatial_linear3_->forward(x);
}

infinicore::Tensor Ernie4_5VariableResolutionResampler::temporal_forward(const infinicore::Tensor &hidden_states,
                                                                         const infinicore::Tensor &grid_thw) const {
    if (temporal_conv_size_ != 2) {
        throw std::runtime_error("Ernie4_5VariableResolutionResampler: only temporal_conv_size=2 is supported");
    }

    const auto grids = read_grid_thw_cpu(grid_thw);
    const size_t spatial_pack = spatial_conv_size_ * spatial_conv_size_;
    size_t spatial_base = 0;
    size_t out_rows = 0;
    for (const auto &grid : grids) {
        if (grid.h % static_cast<int64_t>(spatial_conv_size_) != 0 || grid.w % static_cast<int64_t>(spatial_conv_size_) != 0) {
            throw std::runtime_error("Ernie4_5VariableResolutionResampler: invalid spatial grid");
        }
        if (grid.t > 1 && (grid.t % 2) != 0) {
            throw std::runtime_error("Ernie4_5VariableResolutionResampler: odd video temporal size is not supported");
        }
        const size_t spatial_size = static_cast<size_t>(grid.h * grid.w) / spatial_pack;
        spatial_base += static_cast<size_t>(grid.t) * spatial_size;
        out_rows += ((static_cast<size_t>(grid.t) + 1) / 2) * spatial_size;
    }
    if (spatial_base != hidden_states->size(0)) {
        throw std::runtime_error("Ernie4_5VariableResolutionResampler: grid_thw does not match spatial hidden length");
    }

    auto x = infinicore::Tensor::empty(
        {out_rows, hidden_states->size(1) * 2},
        hidden_states->dtype(),
        hidden_states->device());
    spatial_base = 0;
    size_t out_offset = 0;
    for (const auto &grid : grids) {
        const size_t spatial_size = static_cast<size_t>(grid.h * grid.w) / spatial_pack;
        for (int64_t temp_offset = 0; temp_offset < grid.t; temp_offset += 2) {
            const size_t first = spatial_base + static_cast<size_t>(temp_offset) * spatial_size;
            const size_t second_temp = static_cast<size_t>((grid.t > 1) ? temp_offset + 1 : 0);
            const size_t second = spatial_base + second_temp * spatial_size;
            auto out_slice = x->narrow({{0, out_offset, spatial_size}});
            out_slice->narrow({{1, 0, hidden_states->size(1)}})
                ->copy_from(hidden_states->narrow({{0, first, spatial_size}}));
            out_slice->narrow({{1, hidden_states->size(1), hidden_states->size(1)}})
                ->copy_from(hidden_states->narrow({{0, second, spatial_size}}));
            out_offset += spatial_size;
        }
        spatial_base += static_cast<size_t>(grid.t) * spatial_size;
    }
    x = temporal_linear0_->forward(x);
    x = infinicore::op::gelu(x);
    x = temporal_linear2_->forward(x);
    return temporal_linear3_->forward(x);
}

infinicore::Tensor Ernie4_5VariableResolutionResampler::forward(const infinicore::Tensor &hidden_states,
                                                                const infinicore::Tensor &grid_thw) const {
    auto x = spatial_forward(hidden_states);
    if (use_temporal_conv_) {
        x = temporal_forward(x, grid_thw);
    }
    x = mlp_->forward(x);
    return after_norm_->forward(x);
}

} // namespace infinilm::models::ernie4_5_moe_vl
