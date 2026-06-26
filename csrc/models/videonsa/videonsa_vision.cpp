#include "videonsa_vision.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/take.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace infinilm::models::videonsa {

VideoNSAPatchEmbed::VideoNSAPatchEmbed(const nlohmann::json &config,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device)
    : out_hidden_size_(config.value("hidden_size", 1280)),
      patch_dim_(config.value("in_channels", 3)
                 * config.value("temporal_patch_size", 2)
                 * config.value("patch_size", 14)
                 * config.value("patch_size", 14)) {
    INFINICORE_NN_PARAMETER_INIT(proj_weight, ({out_hidden_size_, patch_dim_}, dtype, device));
    this->register_parameter("proj.weight", proj_weight_);
}

infinicore::Tensor VideoNSAPatchEmbed::forward(const infinicore::Tensor &pixel_values) const {
    auto input = pixel_values->view({pixel_values->size(0), patch_dim_});
    return infinicore::op::linear(input, proj_weight_, std::nullopt);
}

VideoNSAVisionAttention::VideoNSAVisionAttention(const nlohmann::json &config,
                                                 const infinicore::DataType &dtype,
                                                 const infinicore::Device &device)
    : hidden_size_(config.value("hidden_size", 1280)),
      num_heads_(config.value("num_heads", 16)),
      head_dim_(hidden_size_ / num_heads_),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    INFINICORE_NN_MODULE_INIT(qkv, hidden_size_, 3 * hidden_size_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(proj, hidden_size_, hidden_size_, true, dtype, device);
}

infinicore::Tensor VideoNSAVisionAttention::forward(const infinicore::Tensor &hidden_states,
                                                    const std::vector<int64_t> *cu_window_seqlens) const {
    const size_t seq_len = hidden_states->size(0);
    auto hidden = const_cast<infinicore::Tensor &>(hidden_states);
    auto qkv = qkv_->forward(hidden)->view({seq_len, 3, num_heads_, head_dim_});

    auto run_dense = [&](size_t start, size_t len) -> infinicore::Tensor {
        auto q = qkv->narrow({{0, start, len}, {1, 0, 1}})->squeeze(1)->permute({1, 0, 2})->contiguous();
        auto k = qkv->narrow({{0, start, len}, {1, 1, 1}})->squeeze(1)->permute({1, 0, 2})->contiguous();
        auto v = qkv->narrow({{0, start, len}, {1, 2, 1}})->squeeze(1)->permute({1, 0, 2})->contiguous();

        auto q_flat = q->view({num_heads_, len, head_dim_});
        auto k_flat = k->view({num_heads_, len, head_dim_});
        auto v_flat = v->view({num_heads_, len, head_dim_});
        auto attn = infinicore::op::matmul(q_flat, k_flat->permute({0, 2, 1}), scale_);
        infinicore::op::softmax_(attn, attn, -1);
        return infinicore::op::matmul(attn, v_flat)
            ->view({num_heads_, len, head_dim_})
            ->permute({1, 0, 2})
            ->contiguous()
            ->view({len, hidden_size_});
    };

    infinicore::Tensor out;
    if (cu_window_seqlens == nullptr || cu_window_seqlens->size() <= 2) {
        out = run_dense(0, seq_len);
    } else {
        const size_t first_len = static_cast<size_t>((*cu_window_seqlens)[1] - (*cu_window_seqlens)[0]);
        bool uniform_windows = first_len > 0;
        for (size_t i = 1; i + 1 < cu_window_seqlens->size(); ++i) {
            const size_t len = static_cast<size_t>((*cu_window_seqlens)[i + 1] - (*cu_window_seqlens)[i]);
            uniform_windows = uniform_windows && len == first_len;
        }

        if (uniform_windows) {
            const size_t num_windows = cu_window_seqlens->size() - 1;
            auto qkv_windows = qkv->view({num_windows, first_len, 3, num_heads_, head_dim_});
            auto q = qkv_windows->narrow({{2, 0, 1}})->squeeze(2)->permute({0, 2, 1, 3})->contiguous();
            auto k = qkv_windows->narrow({{2, 1, 1}})->squeeze(2)->permute({0, 2, 1, 3})->contiguous();
            auto v = qkv_windows->narrow({{2, 2, 1}})->squeeze(2)->permute({0, 2, 1, 3})->contiguous();
            auto q_flat = q->view({num_windows * num_heads_, first_len, head_dim_});
            auto k_flat = k->view({num_windows * num_heads_, first_len, head_dim_});
            auto v_flat = v->view({num_windows * num_heads_, first_len, head_dim_});
            auto attn = infinicore::op::matmul(q_flat, k_flat->permute({0, 2, 1}), scale_);
            infinicore::op::softmax_(attn, attn, -1);
            out = infinicore::op::matmul(attn, v_flat)
                      ->view({num_windows, num_heads_, first_len, head_dim_})
                      ->permute({0, 2, 1, 3})
                      ->contiguous()
                      ->view({seq_len, hidden_size_});
        } else {
            std::vector<infinicore::Tensor> chunks;
            chunks.reserve(cu_window_seqlens->size() - 1);
            for (size_t i = 0; i + 1 < cu_window_seqlens->size(); ++i) {
                const size_t start = static_cast<size_t>((*cu_window_seqlens)[i]);
                const size_t end = static_cast<size_t>((*cu_window_seqlens)[i + 1]);
                if (end > start) {
                    chunks.push_back(run_dense(start, end - start));
                }
            }
            out = chunks.size() == 1 ? chunks.front() : infinicore::op::cat(chunks, 0);
        }
    }
    return proj_->forward(out);
}

VideoNSAVisionMLP::VideoNSAVisionMLP(const nlohmann::json &config,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device) {
    const size_t hidden_size = config.value("hidden_size", 1280);
    const size_t intermediate_size = config.value("intermediate_size", 3420);
    INFINICORE_NN_MODULE_INIT(gate_proj, hidden_size, intermediate_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(up_proj, hidden_size, intermediate_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size, hidden_size, true, dtype, device);
}

infinicore::Tensor VideoNSAVisionMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden = const_cast<infinicore::Tensor &>(hidden_states);
    auto gate = gate_proj_->forward(hidden);
    auto up = up_proj_->forward(hidden);
    auto x = infinicore::op::swiglu(up, gate);
    return down_proj_->forward(x);
}

VideoNSAVisionBlock::VideoNSAVisionBlock(const nlohmann::json &config,
                                         const infinicore::DataType &dtype,
                                         const infinicore::Device &device) {
    const size_t hidden_size = config.value("hidden_size", 1280);
    INFINICORE_NN_MODULE_INIT(norm1, hidden_size, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, config, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm2, hidden_size, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, config, dtype, device);
}

infinicore::Tensor VideoNSAVisionBlock::forward(const infinicore::Tensor &hidden_states,
                                                const std::vector<int64_t> *cu_window_seqlens) const {
    auto residual = hidden_states;
    auto x = norm1_->forward(hidden_states);
    x = attn_->forward(x, cu_window_seqlens);
    x = infinicore::op::add(x, residual);

    residual = x;
    x = norm2_->forward(x);
    x = mlp_->forward(x);
    return infinicore::op::add(x, residual);
}

VideoNSAPatchMerger::VideoNSAPatchMerger(const nlohmann::json &config,
                                         const infinicore::DataType &dtype,
                                         const infinicore::Device &device)
    : hidden_size_(config.value("hidden_size", 1280)),
      spatial_merge_unit_(config.value("spatial_merge_size", 2) * config.value("spatial_merge_size", 2)),
      merged_size_(hidden_size_ * spatial_merge_unit_) {
    const size_t out_hidden_size = config.value("out_hidden_size", 3584);
    INFINICORE_NN_MODULE_INIT(ln_q, hidden_size_, 1e-6, dtype, device);
    mlp_0_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("mlp.0", merged_size_, merged_size_, true, dtype, device);
    mlp_2_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("mlp.2", merged_size_, out_hidden_size, true, dtype, device);
}

infinicore::Tensor VideoNSAPatchMerger::forward(const infinicore::Tensor &hidden_states) const {
    auto x = ln_q_->forward(hidden_states)->view({hidden_states->size(0) / spatial_merge_unit_, merged_size_});
    x = mlp_0_->forward(x);
    x = infinicore::op::gelu(x);
    return mlp_2_->forward(x);
}

VideoNSAVisionModel::VideoNSAVisionModel(const nlohmann::json &config,
                                         const infinicore::DataType &dtype,
                                         const infinicore::Device &device)
    : depth_(config.value("depth", 32)),
      hidden_size_(config.value("hidden_size", 1280)),
      patch_size_(config.value("patch_size", 14)),
      spatial_merge_size_(config.value("spatial_merge_size", 2)),
      spatial_merge_unit_(spatial_merge_size_ * spatial_merge_size_),
      window_size_(config.value("window_size", 112)) {
    if (config.contains("fullatt_block_indexes")) {
        for (const auto &idx : config["fullatt_block_indexes"]) {
            fullatt_block_indexes_.insert(idx.get<size_t>());
        }
    }
    INFINICORE_NN_MODULE_INIT(patch_embed, config, dtype, device);
    blocks_.reserve(depth_);
    for (size_t i = 0; i < depth_; ++i) {
        blocks_.push_back(this->register_module<VideoNSAVisionBlock>("blocks." + std::to_string(i), config, dtype, device));
    }
    INFINICORE_NN_MODULE_INIT(merger, config, dtype, device);
}

VideoNSAVisionModel::WindowMetadata VideoNSAVisionModel::build_window_metadata_(const infinicore::Tensor &grid_thw) const {
    auto grid_cpu = grid_thw->to(infinicore::Device::cpu());
    const int64_t *grid = reinterpret_cast<const int64_t *>(grid_cpu->data());
    const size_t n = grid_cpu->size(0);
    const int64_t vit_window = static_cast<int64_t>(window_size_ / spatial_merge_size_ / patch_size_);
    if (vit_window <= 0) {
        throw std::runtime_error("VideoNSAVisionModel: invalid vision window size");
    }

    WindowMetadata meta;
    meta.cu_window_seqlens.push_back(0);
    meta.cu_full_seqlens.push_back(0);
    int64_t merged_offset = 0;
    for (size_t item = 0; item < n; ++item) {
        const int64_t grid_t = grid[item * 3];
        const int64_t grid_h = grid[item * 3 + 1];
        const int64_t grid_w = grid[item * 3 + 2];
        const int64_t llm_h = grid_h / static_cast<int64_t>(spatial_merge_size_);
        const int64_t llm_w = grid_w / static_cast<int64_t>(spatial_merge_size_);
        const int64_t pad_h = (vit_window - llm_h % vit_window) % vit_window;
        const int64_t pad_w = (vit_window - llm_w % vit_window) % vit_window;
        const int64_t num_win_h = (llm_h + pad_h) / vit_window;
        const int64_t num_win_w = (llm_w + pad_w) / vit_window;

        for (int64_t t = 0; t < grid_t; ++t) {
            meta.cu_full_seqlens.push_back(meta.cu_full_seqlens.back() + llm_h * llm_w * static_cast<int64_t>(spatial_merge_unit_));
            for (int64_t wh = 0; wh < num_win_h; ++wh) {
                for (int64_t ww = 0; ww < num_win_w; ++ww) {
                    int64_t merged_in_window = 0;
                    for (int64_t ih = 0; ih < vit_window; ++ih) {
                        const int64_t h = wh * vit_window + ih;
                        if (h >= llm_h) {
                            continue;
                        }
                        for (int64_t iw = 0; iw < vit_window; ++iw) {
                            const int64_t w = ww * vit_window + iw;
                            if (w >= llm_w) {
                                continue;
                            }
                            const int64_t merged_idx = merged_offset + t * llm_h * llm_w + h * llm_w + w;
                            for (size_t u = 0; u < spatial_merge_unit_; ++u) {
                                meta.patch_order.push_back(merged_idx * static_cast<int64_t>(spatial_merge_unit_) + static_cast<int64_t>(u));
                            }
                            ++merged_in_window;
                        }
                    }
                    meta.cu_window_seqlens.push_back(meta.cu_window_seqlens.back() + merged_in_window * static_cast<int64_t>(spatial_merge_unit_));
                }
            }
        }
        merged_offset += grid_t * llm_h * llm_w;
    }

    std::vector<int64_t> window_merged_order;
    window_merged_order.reserve(meta.patch_order.size() / spatial_merge_unit_);
    for (size_t i = 0; i < meta.patch_order.size(); i += spatial_merge_unit_) {
        window_merged_order.push_back(meta.patch_order[i] / static_cast<int64_t>(spatial_merge_unit_));
    }
    meta.reverse_order.resize(window_merged_order.size());
    std::iota(meta.reverse_order.begin(), meta.reverse_order.end(), 0);
    std::sort(meta.reverse_order.begin(), meta.reverse_order.end(), [&](int64_t a, int64_t b) {
        return window_merged_order[a] < window_merged_order[b];
    });
    return meta;
}

infinicore::Tensor VideoNSAVisionModel::gather_rows_(const infinicore::Tensor &hidden_states,
                                                     const std::vector<int64_t> &row_order) const {
    const size_t rows = row_order.size();
    const size_t width = hidden_states->size(1);
    std::vector<int64_t> flat_indices(rows * width);
    for (size_t r = 0; r < rows; ++r) {
        const int64_t src_row = row_order[r];
        for (size_t c = 0; c < width; ++c) {
            flat_indices[r * width + c] = src_row * static_cast<int64_t>(width) + static_cast<int64_t>(c);
        }
    }
    auto indices = infinicore::Tensor::empty({rows, width}, infinicore::DataType::I64, hidden_states->device());
    infinicore::context::memcpyH2D(indices->data(), flat_indices.data(), flat_indices.size() * sizeof(int64_t), false);
    return infinicore::op::take(hidden_states->contiguous(), indices);
}

infinicore::Tensor VideoNSAVisionModel::forward(const infinicore::Tensor &pixel_values,
                                                const infinicore::Tensor &grid_thw) const {
    auto hidden_states = patch_embed_->forward(pixel_values);
    auto window_meta = build_window_metadata_(grid_thw);
    if (window_meta.patch_order.size() != hidden_states->size(0)) {
        throw std::runtime_error("VideoNSAVisionModel: window metadata does not match patch sequence length");
    }
    hidden_states = gather_rows_(hidden_states, window_meta.patch_order);
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const auto *cu_segments = fullatt_block_indexes_.count(i) > 0 ? &window_meta.cu_full_seqlens : &window_meta.cu_window_seqlens;
        hidden_states = blocks_[i]->forward(hidden_states, cu_segments);
    }
    auto merged = merger_->forward(hidden_states);
    return gather_rows_(merged, window_meta.reverse_order);
}

} // namespace infinilm::models::videonsa
