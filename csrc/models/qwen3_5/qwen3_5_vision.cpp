#include "qwen3_5_vision.hpp"

#include "../../utils.hpp"

#include <infinicore/ops.hpp>
#include <infinicore/ops/mha.hpp>
#include <infinicore/ops/upsample_bilinear.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::models::qwen3_5 {
namespace {

size_t get_size_or_first(const nlohmann::json &config, const char *key, size_t default_value) {
    if (!config.contains(key) || config.at(key).is_null()) {
        return default_value;
    }
    const auto &value = config.at(key);
    if (value.is_array()) {
        return value.empty() ? default_value : value.at(0).get<size_t>();
    }
    return value.get<size_t>();
}

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
    throw std::runtime_error("Qwen35VisionModel: grid_thw must be int32 or int64");
}

} // namespace

Qwen35VisionPatchProj::Qwen35VisionPatchProj(size_t in_channels,
                                             size_t hidden_size,
                                             size_t temporal_patch_size,
                                             size_t patch_size,
                                             const infinicore::DataType &dtype,
                                             const infinicore::Device &device)
    : in_channels_(in_channels),
      hidden_size_(hidden_size),
      temporal_patch_size_(temporal_patch_size),
      patch_size_(patch_size) {
    INFINICORE_NN_PARAMETER_INIT(weight, ({hidden_size_, in_channels_, temporal_patch_size_, patch_size_, patch_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(bias, ({hidden_size_}, dtype, device));
}

infinicore::Tensor Qwen35VisionPatchProj::forward(const infinicore::Tensor &hidden_states) const {
    const size_t patch_dim = in_channels_ * temporal_patch_size_ * patch_size_ * patch_size_;
    if (hidden_states->shape().size() != 2 || hidden_states->size(1) != patch_dim) {
        throw std::runtime_error("Qwen35VisionPatchProj: expected pixel_values shape [num_patches, patch_dim]");
    }
    auto weight_2d = weight_->view({hidden_size_, patch_dim});
    return infinicore::op::linear(hidden_states, weight_2d, std::make_optional<infinicore::Tensor>(bias_));
}

Qwen35VisionPatchEmbed::Qwen35VisionPatchEmbed(const nlohmann::json &config,
                                               const infinicore::DataType &dtype,
                                               const infinicore::Device &device) {
    const size_t in_channels = config.value("in_channels", 3);
    const size_t hidden_size = config.value("hidden_size", 1152);
    const size_t temporal_patch_size = get_size_or_first(config, "temporal_patch_size", 2);
    const size_t patch_size = get_size_or_first(config, "patch_size", 16);
    INFINICORE_NN_MODULE_INIT(proj, in_channels, hidden_size, temporal_patch_size, patch_size, dtype, device);
}

infinicore::Tensor Qwen35VisionPatchEmbed::forward(const infinicore::Tensor &hidden_states) const {
    return proj_->forward(hidden_states);
}

Qwen35VisionAttention::Qwen35VisionAttention(const nlohmann::json &config,
                                             const infinicore::DataType &dtype,
                                             const infinicore::Device &device)
    : hidden_size_(config.value("hidden_size", 1152)),
      num_heads_(config.value("num_heads", 16)),
      head_dim_(hidden_size_ / num_heads_),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    if (hidden_size_ % num_heads_ != 0) {
        throw std::runtime_error("Qwen35VisionAttention: hidden_size must be divisible by num_heads");
    }
    if (head_dim_ % 4 != 0) {
        throw std::runtime_error("Qwen35VisionAttention: head_dim must be divisible by 4 for 2D RoPE");
    }
    const size_t axis_head_dim = head_dim_ / 2;
    INFINICORE_NN_MODULE_INIT(rotary_emb, axis_head_dim, axis_head_dim, 8192, 10000.0, infinicore::nn::RoPE::Algo::GPT_NEOX, dtype, device);
    INFINICORE_NN_MODULE_INIT(qkv, hidden_size_, hidden_size_ * 3, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(proj, hidden_size_, hidden_size_, true, dtype, device);
}

infinicore::Tensor Qwen35VisionAttention::forward(const infinicore::Tensor &hidden_states,
                                                  const infinicore::Tensor &row_position_ids,
                                                  const infinicore::Tensor &col_position_ids) const {
    const size_t seq_len = hidden_states->size(0);
    const size_t axis_head_dim = head_dim_ / 2;
    auto hidden_mut = hidden_states;
    auto qkv = qkv_->forward(hidden_mut)->view({seq_len, 3, num_heads_, head_dim_});
    auto q = qkv->narrow({{1, 0, 1}})->squeeze(1)->contiguous();
    auto k = qkv->narrow({{1, 1, 1}})->squeeze(1)->contiguous();
    auto v = qkv->narrow({{1, 2, 1}})->squeeze(1)->contiguous()->unsqueeze(0);

    const size_t axis_pair_dim = axis_head_dim / 2;
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

Qwen35VisionMLP::Qwen35VisionMLP(const nlohmann::json &config,
                                 const infinicore::DataType &dtype,
                                 const infinicore::Device &device)
    : activation_(config.value("hidden_act", "gelu_pytorch_tanh")) {
    const size_t hidden_size = config.value("hidden_size", 1152);
    const size_t intermediate_size = config.value("intermediate_size", 4304);
    INFINICORE_NN_MODULE_INIT(linear_fc1, hidden_size, intermediate_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear_fc2, intermediate_size, hidden_size, true, dtype, device);
}

infinicore::Tensor Qwen35VisionMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden_mut = hidden_states;
    auto x = linear_fc1_->forward(hidden_mut);
    if (activation_ == "gelu" || activation_ == "gelu_pytorch_tanh") {
        x = activation_ == "gelu" ? infinicore::op::gelu(x) : infinicore::op::gelu_tanh(x);
    } else {
        throw std::runtime_error("Qwen35VisionMLP: unsupported activation " + activation_);
    }
    return linear_fc2_->forward(x);
}

Qwen35VisionBlock::Qwen35VisionBlock(const nlohmann::json &config,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device) {
    const size_t hidden_size = config.value("hidden_size", 1152);
    const double norm_eps = config.value("layer_norm_eps", config.value("rms_norm_eps", 1e-6));
    INFINICORE_NN_MODULE_INIT(norm1, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm2, hidden_size, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, config, dtype, device);
}

infinicore::Tensor Qwen35VisionBlock::forward(const infinicore::Tensor &hidden_states,
                                              const infinicore::Tensor &row_position_ids,
                                              const infinicore::Tensor &col_position_ids) const {
    auto x = norm1_->forward(hidden_states);
    x = attn_->forward(x, row_position_ids, col_position_ids);
    x = infinicore::op::add(x, hidden_states);
    auto residual = x;
    x = norm2_->forward(x);
    x = mlp_->forward(x);
    return infinicore::op::add(x, residual);
}

Qwen35VisionPatchMerger::Qwen35VisionPatchMerger(const nlohmann::json &config,
                                                 const infinicore::DataType &dtype,
                                                 const infinicore::Device &device)
    : hidden_size_(config.value("hidden_size", 1152)),
      merged_size_(hidden_size_ * config.value("spatial_merge_size", 2) * config.value("spatial_merge_size", 2)) {
    const size_t out_hidden_size = config.value("out_hidden_size", hidden_size_);
    const double norm_eps = config.value("layer_norm_eps", config.value("rms_norm_eps", 1e-6));
    INFINICORE_NN_MODULE_INIT(norm, hidden_size_, norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear_fc1, merged_size_, merged_size_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(linear_fc2, merged_size_, out_hidden_size, true, dtype, device);
}

infinicore::Tensor Qwen35VisionPatchMerger::forward(const infinicore::Tensor &hidden_states) const {
    auto x = norm_->forward(hidden_states)->view({hidden_states->size(0) / (merged_size_ / hidden_size_), merged_size_});
    x = linear_fc1_->forward(x);
    x = infinicore::op::gelu(x);
    return linear_fc2_->forward(x);
}

Qwen35VisionModel::Qwen35VisionModel(const nlohmann::json &config,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device)
    : hidden_size_(config.value("hidden_size", 1152)),
      num_heads_(config.value("num_heads", 16)),
      head_dim_(hidden_size_ / num_heads_),
      spatial_merge_size_(config.value("spatial_merge_size", 2)),
      num_grid_per_side_(static_cast<size_t>(std::sqrt(static_cast<double>(config.value("num_position_embeddings", 2304))))) {
    const size_t num_position_embeddings = config.value("num_position_embeddings", 2304);
    const size_t depth = config.value("depth", config.value("num_hidden_layers", 27));

    INFINICORE_NN_MODULE_INIT(patch_embed, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(pos_embed, num_position_embeddings, hidden_size_, std::nullopt, dtype, device);
    blocks_.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        blocks_.push_back(this->register_module<Qwen35VisionBlock>("blocks." + std::to_string(i), config, dtype, device));
    }
    INFINICORE_NN_MODULE_INIT(merger, config, dtype, device);
}

infinicore::Tensor Qwen35VisionModel::fast_pos_embed_interpolate(const infinicore::Tensor &image_grid_thw) const {
    auto grid = tensor_to_i64_vector(image_grid_thw);
    if (grid.size() != 3) {
        throw std::runtime_error("Qwen35VisionModel: image_grid_thw must have shape [3]");
    }
    const size_t grid_t = static_cast<size_t>(grid[0]);
    const size_t grid_h = static_cast<size_t>(grid[1]);
    const size_t grid_w = static_cast<size_t>(grid[2]);
    if (grid_h % spatial_merge_size_ != 0 || grid_w % spatial_merge_size_ != 0) {
        throw std::runtime_error("Qwen35VisionModel: grid_h and grid_w must be divisible by spatial_merge_size");
    }

    auto pos_nchw = pos_embed_->weight()
                        ->view({num_grid_per_side_, num_grid_per_side_, hidden_size_})
                        ->permute({2, 0, 1})
                        ->unsqueeze(0);
    auto resized = infinicore::op::upsample_bilinear(
        pos_nchw,
        {static_cast<int64_t>(grid_h), static_cast<int64_t>(grid_w)},
        true);
    auto one_frame = resized->squeeze(0)
                         ->permute({1, 2, 0})
                         ->view({1, grid_h / spatial_merge_size_, spatial_merge_size_, grid_w / spatial_merge_size_, spatial_merge_size_, hidden_size_})
                         ->permute({0, 1, 3, 2, 4, 5})
                         ->contiguous()
                         ->view({grid_h * grid_w, hidden_size_});
    if (grid_t == 1) {
        return one_frame;
    }

    auto pos_embeds = infinicore::Tensor::empty({grid_t * grid_h * grid_w, hidden_size_}, one_frame->dtype(), one_frame->device());
    for (size_t t = 0; t < grid_t; ++t) {
        pos_embeds->narrow({{0, t * grid_h * grid_w, grid_h * grid_w}})->copy_from(one_frame);
    }
    return pos_embeds;
}

infinicore::Tensor Qwen35VisionModel::build_rotary_position_ids(const infinicore::Tensor &image_grid_thw) const {
    auto grid = tensor_to_i64_vector(image_grid_thw);
    if (grid.size() % 3 != 0) {
        throw std::runtime_error("Qwen35VisionModel: image_grid_thw must have shape [3] or [num_images, 3]");
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
            throw std::runtime_error("Qwen35VisionModel: grid_h and grid_w must be divisible by spatial_merge_size");
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
    return position_ids_cpu->to(image_grid_thw->device());
}

infinicore::Tensor Qwen35VisionModel::forward(const infinicore::Tensor &pixel_values,
                                              const infinicore::Tensor &image_grid_thw) const {
    auto hidden_states = patch_embed_->forward(pixel_values);
    auto pos_embeds = fast_pos_embed_interpolate(image_grid_thw);
    hidden_states = infinicore::op::add(hidden_states, pos_embeds);

    auto position_ids = build_rotary_position_ids(image_grid_thw);
    auto row_position_ids = position_ids->narrow({{0, 0, 1}})->view({position_ids->size(1)});
    auto col_position_ids = position_ids->narrow({{0, 1, 1}})->view({position_ids->size(1)});

    for (const auto &block : blocks_) {
        hidden_states = block->forward(hidden_states, row_position_ids, col_position_ids);
    }
    return merger_->forward(hidden_states);
}

} // namespace infinilm::models::qwen3_5
