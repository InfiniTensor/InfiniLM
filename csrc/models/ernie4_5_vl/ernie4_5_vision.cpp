#include "ernie4_5_vision.hpp"

#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/mha.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::ernie4_5_vl {
namespace {

std::vector<int64_t> tensor_to_i64_vector(const infinicore::Tensor &tensor) {
    auto cpu_tensor = tensor->to(infinicore::Device::cpu());
    std::vector<int64_t> values(cpu_tensor->numel());
    if (cpu_tensor->dtype() == infinicore::DataType::I64) {
        const auto *ptr = reinterpret_cast<const int64_t *>(cpu_tensor->data());
        values.assign(ptr, ptr + cpu_tensor->numel());
        return values;
    }
    if (cpu_tensor->dtype() == infinicore::DataType::I32) {
        const auto *ptr = reinterpret_cast<const int32_t *>(cpu_tensor->data());
        for (size_t i = 0; i < cpu_tensor->numel(); ++i) {
            values[i] = static_cast<int64_t>(ptr[i]);
        }
        return values;
    }
    throw std::runtime_error("ERNIE 4.5 VL grid_thw must be int32 or int64");
}

infinicore::Tensor cat_or_single(std::vector<infinicore::Tensor> tensors, int dim) {
    if (tensors.empty()) {
        throw std::runtime_error("ERNIE 4.5 VL internal cat received no tensors");
    }
    if (tensors.size() == 1) {
        return tensors[0];
    }
    return infinicore::op::cat(std::move(tensors), dim);
}

} // namespace

Ernie45VisionLayerNorm::Ernie45VisionLayerNorm(size_t normalized_shape,
                                               double eps,
                                               const infinicore::DataType &dtype,
                                               const infinicore::Device &device)
    : eps_(eps) {
    INFINICORE_NN_PARAMETER_INIT(weight, ({normalized_shape}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(bias, ({normalized_shape}, dtype, device));
}

infinicore::Tensor Ernie45VisionLayerNorm::forward(const infinicore::Tensor &hidden_states) const {
    return infinicore::op::layer_norm(hidden_states, weight_, bias_, static_cast<float>(eps_));
}

Ernie45VisionPatchEmbed::Ernie45VisionPatchEmbed(const nlohmann::json &config,
                                                 const infinicore::DataType &dtype,
                                                 const infinicore::Device &device) {
    const size_t in_channels = config.value("in_channels", config.value("in_chans", 3));
    const size_t patch_size = config.value("patch_size", config.value("spatial_patch_size", 14));
    const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
    INFINICORE_NN_MODULE_INIT(proj, in_channels * patch_size * patch_size, embed_dim, false, dtype, device);
}

infinicore::Tensor Ernie45VisionPatchEmbed::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    return proj_->forward(hidden_states_mutable);
}

Ernie45VisionAttention::Ernie45VisionAttention(const nlohmann::json &config,
                                               const infinicore::DataType &dtype,
                                               const infinicore::Device &device) {
    hidden_size_ = config.value("embed_dim", config.value("hidden_size", 1280));
    num_heads_ = config.value("num_heads", 16);
    if (hidden_size_ % num_heads_ != 0) {
        throw std::runtime_error("Ernie45VisionAttention: hidden_size must be divisible by num_heads");
    }
    head_dim_ = hidden_size_ / num_heads_;
    if (head_dim_ % 4 != 0) {
        throw std::runtime_error("Ernie45VisionAttention: head_dim must be divisible by 4 for 2D RoPE");
    }
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    const size_t axis_head_dim = head_dim_ / 2;
    INFINICORE_NN_MODULE_INIT(rotary_emb, axis_head_dim, axis_head_dim, 8192, 10000.0, infinicore::nn::RoPE::Algo::GPT_NEOX, dtype, device);
    INFINICORE_NN_MODULE_INIT(qkv, hidden_size_, hidden_size_ * 3, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(proj, hidden_size_, hidden_size_, true, dtype, device);
}

infinicore::Tensor Ernie45VisionAttention::forward(const infinicore::Tensor &hidden_states,
                                                   const infinicore::Tensor &row_position_ids,
                                                   const infinicore::Tensor &col_position_ids) const {
    const size_t seq_len = hidden_states->size(0);
    const size_t axis_head_dim = head_dim_ / 2;
    const size_t axis_pair_dim = axis_head_dim / 2;

    auto hidden_mut = hidden_states;
    auto qkv = qkv_->forward(hidden_mut)->view({seq_len, 3, num_heads_, head_dim_});
    auto q = qkv->narrow({{1, 0, 1}})->squeeze(1)->contiguous();
    auto k = qkv->narrow({{1, 1, 1}})->squeeze(1)->contiguous();
    auto v = qkv->narrow({{1, 2, 1}})->squeeze(1)->contiguous()->unsqueeze(0);

    auto apply_2d_rope = [&](const infinicore::Tensor &x) {
        auto row = infinicore::Tensor::empty({seq_len, num_heads_, axis_head_dim}, x->dtype(), x->device());
        auto col = infinicore::Tensor::empty({seq_len, num_heads_, axis_head_dim}, x->dtype(), x->device());
        row->narrow({{2, 0, axis_pair_dim}})->copy_from(x->narrow({{2, 0, axis_pair_dim}}));
        row->narrow({{2, axis_pair_dim, axis_pair_dim}})->copy_from(x->narrow({{2, axis_head_dim, axis_pair_dim}}));
        col->narrow({{2, 0, axis_pair_dim}})->copy_from(x->narrow({{2, axis_pair_dim, axis_pair_dim}}));
        col->narrow({{2, axis_pair_dim, axis_pair_dim}})->copy_from(x->narrow({{2, axis_head_dim + axis_pair_dim, axis_pair_dim}}));

        rotary_emb_->forward(row, row_position_ids, true);
        rotary_emb_->forward(col, col_position_ids, true);

        x->narrow({{2, 0, axis_pair_dim}})->copy_from(row->narrow({{2, 0, axis_pair_dim}}));
        x->narrow({{2, axis_head_dim, axis_pair_dim}})->copy_from(row->narrow({{2, axis_pair_dim, axis_pair_dim}}));
        x->narrow({{2, axis_pair_dim, axis_pair_dim}})->copy_from(col->narrow({{2, 0, axis_pair_dim}}));
        x->narrow({{2, axis_head_dim + axis_pair_dim, axis_pair_dim}})->copy_from(col->narrow({{2, axis_pair_dim, axis_pair_dim}}));
    };
    apply_2d_rope(q);
    apply_2d_rope(k);

    auto out = infinicore::op::mha(q->unsqueeze(0), k->unsqueeze(0), v, std::nullopt, scale_, false)->view({seq_len, hidden_size_});
    return proj_->forward(out);
}

Ernie45VisionMLP::Ernie45VisionMLP(const nlohmann::json &config,
                                   const infinicore::DataType &dtype,
                                   const infinicore::Device &device) {
    const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
    const size_t hidden_dim = static_cast<size_t>(static_cast<double>(embed_dim) * config.value("mlp_ratio", 4.0));
    INFINICORE_NN_MODULE_INIT(fc1, embed_dim, hidden_dim, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(fc2, hidden_dim, embed_dim, true, dtype, device);
}

infinicore::Tensor Ernie45VisionMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto x = fc1_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    x = infinicore::op::quick_gelu(x);
    return fc2_->forward(x);
}

Ernie45VisionBlock::Ernie45VisionBlock(const nlohmann::json &config,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device) {
    const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
    INFINICORE_NN_MODULE_INIT(norm1, embed_dim, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm2, embed_dim, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, config, dtype, device);
}

infinicore::Tensor Ernie45VisionBlock::forward(const infinicore::Tensor &hidden_states,
                                               const infinicore::Tensor &row_position_ids,
                                               const infinicore::Tensor &col_position_ids) const {
    auto attn_out = attn_->forward(norm1_->forward(hidden_states), row_position_ids, col_position_ids);
    auto hidden_after_attn = infinicore::op::add(hidden_states, attn_out);
    auto mlp_out = mlp_->forward(norm2_->forward(hidden_after_attn));
    return infinicore::op::add(hidden_after_attn, mlp_out);
}

Ernie45VisionModel::Ernie45VisionModel(const nlohmann::json &config,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device) {
    const size_t depth = config.value("depth", 32);
    const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
    spatial_merge_size_ = config.value("spatial_merge_size", 2);
    INFINICORE_NN_MODULE_INIT(patch_embed, config, dtype, device);
    blocks_.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        blocks_.push_back(this->register_module<Ernie45VisionBlock>("blocks." + std::to_string(i), config, dtype, device));
    }
    INFINICORE_NN_MODULE_INIT(ln, embed_dim, 1e-6, dtype, device);
}

infinicore::Tensor Ernie45VisionModel::build_rotary_position_ids(const infinicore::Tensor &grid_thw) const {
    auto grid = tensor_to_i64_vector(grid_thw);
    if (grid.size() % 3 != 0) {
        throw std::runtime_error("Ernie45VisionModel: grid_thw must have shape [3] or [num_images, 3]");
    }

    size_t total_tokens = 0;
    for (size_t i = 0; i < grid.size(); i += 3) {
        total_tokens += static_cast<size_t>(grid[i]) * static_cast<size_t>(grid[i + 1]) * static_cast<size_t>(grid[i + 2]);
    }

    auto position_ids_cpu = infinicore::Tensor::empty({2, total_tokens}, infinicore::DataType::I64, infinicore::Device::cpu());
    auto *position_ids = reinterpret_cast<int64_t *>(position_ids_cpu->data());
    size_t out_token = 0;
    for (size_t i = 0; i < grid.size(); i += 3) {
        const size_t grid_t = static_cast<size_t>(grid[i]);
        const size_t grid_h = static_cast<size_t>(grid[i + 1]);
        const size_t grid_w = static_cast<size_t>(grid[i + 2]);
        if (grid_h % spatial_merge_size_ != 0 || grid_w % spatial_merge_size_ != 0) {
            throw std::runtime_error("Ernie45VisionModel: grid_h and grid_w must be divisible by spatial_merge_size");
        }
        const size_t merged_h = grid_h / spatial_merge_size_;
        const size_t merged_w = grid_w / spatial_merge_size_;
        for (size_t t = 0; t < grid_t; ++t) {
            (void)t;
            for (size_t bh = 0; bh < merged_h; ++bh) {
                for (size_t bw = 0; bw < merged_w; ++bw) {
                    for (size_t ih = 0; ih < spatial_merge_size_; ++ih) {
                        const size_t row = bh * spatial_merge_size_ + ih;
                        for (size_t iw = 0; iw < spatial_merge_size_; ++iw) {
                            const size_t col = bw * spatial_merge_size_ + iw;
                            position_ids[out_token] = static_cast<int64_t>(row);
                            position_ids[total_tokens + out_token] = static_cast<int64_t>(col);
                            ++out_token;
                        }
                    }
                }
            }
        }
    }
    return position_ids_cpu->to(grid_thw->device());
}

infinicore::Tensor Ernie45VisionModel::forward(const infinicore::Tensor &pixel_values,
                                               const infinicore::Tensor &grid_thw) const {
    auto hidden_states = patch_embed_->forward(pixel_values);
    auto position_ids = build_rotary_position_ids(grid_thw);
    auto row_position_ids = position_ids->narrow({{0, 0, 1}})->view({position_ids->size(1)});
    auto col_position_ids = position_ids->narrow({{0, 1, 1}})->view({position_ids->size(1)});

    for (auto &block : blocks_) {
        hidden_states = block->forward(hidden_states, row_position_ids, col_position_ids);
    }
    return ln_->forward(hidden_states);
}

Ernie45ResamplerModel::Ernie45ResamplerModel(const nlohmann::json &config,
                                             const infinicore::DataType &dtype,
                                             const infinicore::Device &device) {
    const size_t pixel_hidden_size = config.value("pixel_hidden_size", 1280);
    const size_t hidden_size = config.value("hidden_size", 2560);
    spatial_conv_size_ = config.value("spatial_conv_size", 2);
    temporal_conv_size_ = config.value("temporal_conv_size", 2);
    const size_t spatial_dim = pixel_hidden_size * spatial_conv_size_ * spatial_conv_size_;
    const size_t temporal_dim = spatial_dim * temporal_conv_size_;
    use_temporal_conv_ = config.value("use_temporal_conv", true);

    spatial_linear_0_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("spatial_linear.0", spatial_dim, spatial_dim, true, dtype, device);
    spatial_linear_2_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("spatial_linear.2", spatial_dim, spatial_dim, true, dtype, device);
    spatial_linear_3_ = this->register_module<Ernie45VisionLayerNorm>("spatial_linear.3", spatial_dim, 1e-6, dtype, device);
    if (use_temporal_conv_) {
        temporal_linear_0_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("temporal_linear.0", temporal_dim, spatial_dim, true, dtype, device);
        temporal_linear_2_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("temporal_linear.2", spatial_dim, spatial_dim, true, dtype, device);
        temporal_linear_3_ = this->register_module<Ernie45VisionLayerNorm>("temporal_linear.3", spatial_dim, 1e-6, dtype, device);
    }
    INFINICORE_NN_MODULE_INIT(mlp, spatial_dim, hidden_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(after_norm, hidden_size, config.value("rms_norm_eps", 1e-5), dtype, device);
}

infinicore::Tensor Ernie45ResamplerModel::forward(const infinicore::Tensor &image_features,
                                                  const infinicore::Tensor &grid_thw) const {
    auto grid = tensor_to_i64_vector(grid_thw);
    if (grid.size() != 3) {
        throw std::runtime_error("Ernie45ResamplerModel: this fallback expects one image grid [3]");
    }
    const size_t grid_t = static_cast<size_t>(grid[0]);
    const size_t grid_h = static_cast<size_t>(grid[1]);
    const size_t grid_w = static_cast<size_t>(grid[2]);
    const size_t spatial_group = spatial_conv_size_ * spatial_conv_size_;
    if (grid_h % spatial_conv_size_ != 0 || grid_w % spatial_conv_size_ != 0) {
        throw std::runtime_error("Ernie45ResamplerModel: grid_h/grid_w must be divisible by spatial_conv_size");
    }
    if (image_features->size(0) != grid_t * grid_h * grid_w) {
        throw std::runtime_error("Ernie45ResamplerModel: image feature length does not match grid_thw");
    }

    auto x = image_features->view({image_features->size(0) / spatial_group, image_features->size(1) * spatial_group});
    x = spatial_linear_0_->forward(x);
    x = infinicore::op::gelu(x);
    x = spatial_linear_2_->forward(x);
    x = spatial_linear_3_->forward(x);

    if (use_temporal_conv_) {
        if (temporal_conv_size_ != 2) {
            throw std::runtime_error("Ernie45ResamplerModel: temporary temporal packing fallback only supports temporal_conv_size=2");
        }
        const size_t spatial_tokens = (grid_h * grid_w) / spatial_group;
        if (x->size(0) != grid_t * spatial_tokens) {
            throw std::runtime_error("Ernie45ResamplerModel: spatial token length mismatch");
        }

        std::vector<infinicore::Tensor> first_frames;
        std::vector<infinicore::Tensor> second_frames;
        for (size_t t = 0; t < grid_t; t += 2) {
            const size_t first = t;
            const size_t second = (t + 1 < grid_t) ? (t + 1) : t;
            first_frames.push_back(x->narrow({{0, first * spatial_tokens, spatial_tokens}}));
            second_frames.push_back(x->narrow({{0, second * spatial_tokens, spatial_tokens}}));
        }
        auto x_first = cat_or_single(std::move(first_frames), 0);
        auto x_second = cat_or_single(std::move(second_frames), 0);
        x = infinicore::op::cat(std::vector<infinicore::Tensor>{x_first, x_second}, 1);
        x = temporal_linear_0_->forward(x);
        x = infinicore::op::gelu(x);
        x = temporal_linear_2_->forward(x);
        x = temporal_linear_3_->forward(x);
    }

    x = mlp_->forward(x);
    return after_norm_->forward(x);
}

} // namespace infinilm::models::ernie4_5_vl
