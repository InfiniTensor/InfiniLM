#include "videonsa_vision.hpp"
#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/mlp/fused_activation.hpp"
#include "../../layers/linear/dense.hpp"
#include "../../execution/tensor_expressions.hpp"

#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"
#include <exception>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace infinilm::models::videonsa {

VideoNSAPatchEmbed::VideoNSAPatchEmbed(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                       const infinilm::runtime::DataType &dtype)
    : out_hidden_size_(config.value("hidden_size", 1280)),
      patch_dim_(config.value("in_channels", 3)
                 * config.value("temporal_patch_size", 2)
                 * config.value("patch_size", 14)
                 * config.value("patch_size", 14)) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("VideoNSAPatchEmbed construction requires eager Runtime");
    try {
    proj_weight_ = infinilm::weights::Weight(runtime, {out_hidden_size_, patch_dim_}, dtype); this->bind_weight("proj_weight", proj_weight_);
    this->bind_weight("proj.weight", proj_weight_);

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor VideoNSAPatchEmbed::forward(infinilm::runtime::Runtime &runtime,
                                                const infinilm::runtime::Tensor &pixel_values) const {
    auto input = pixel_values->view({pixel_values->size(0), patch_dim_});
    return infinilm::layers::linear::dense_projection(runtime, input, proj_weight_.tensor(), std::nullopt);
}

VideoNSAVisionAttention::VideoNSAVisionAttention(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                                 const infinilm::runtime::DataType &dtype)
    : hidden_size_(config.value("hidden_size", 1280)),
      num_heads_(config.value("num_heads", 16)),
      head_dim_(num_heads_ ? hidden_size_ / num_heads_ : 0),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("VideoNSAVisionAttention construction requires eager Runtime");
    if (hidden_size_ == 0 || num_heads_ == 0 || hidden_size_ % num_heads_ != 0)
        throw std::invalid_argument("VideoNSAVisionAttention requires valid head geometry");
    try {
    qkv_ = this->bind_child("qkv", std::make_shared<std::remove_reference_t<decltype(*qkv_ )>>(runtime, hidden_size_, 3 * hidden_size_, true, dtype));
    proj_ = this->bind_child("proj", std::make_shared<std::remove_reference_t<decltype(*proj_ )>>(runtime, hidden_size_, hidden_size_, true, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor VideoNSAVisionAttention::forward(infinilm::runtime::Runtime &runtime,
                                                const infinilm::runtime::Tensor &hidden_states,
                                                    const std::vector<int64_t> *cu_window_seqlens) const {
    const size_t seq_len = hidden_states->size(0);
    auto hidden = const_cast<infinilm::runtime::Tensor &>(hidden_states);
    auto qkv = qkv_->forward(runtime, hidden)->view({seq_len, 3, num_heads_, head_dim_});

    auto run_dense = [&](size_t start, size_t len) -> infinilm::runtime::Tensor {
        auto q = qkv->narrow({{0, start, len}, {1, 0, 1}})->squeeze(1)->permute({1, 0, 2})->contiguous(runtime);
        auto k = qkv->narrow({{0, start, len}, {1, 1, 1}})->squeeze(1)->permute({1, 0, 2})->contiguous(runtime);
        auto v = qkv->narrow({{0, start, len}, {1, 2, 1}})->squeeze(1)->permute({1, 0, 2})->contiguous(runtime);

        auto q_flat = q->view({num_heads_, len, head_dim_});
        auto k_flat = k->view({num_heads_, len, head_dim_});
        auto v_flat = v->view({num_heads_, len, head_dim_});
        const auto key_transposed = k_flat->permute({0, 2, 1});
        auto attn_shape = q_flat->shape();
        attn_shape.back() = key_transposed->size(q_flat->ndim() - 1);
        auto attn = infinilm::runtime::Tensor::empty(runtime, attn_shape, q_flat->dtype());
        infinilm::execution::submit<infini::ops::Gemm>(
            runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
            infini::ops::Gemm::DefaultConfig(runtime.device().type()), q_flat, key_transposed,
            std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{scale_},
            std::optional<float>{0.0f}, std::optional<int>{}, std::optional<int>{}, attn);
        INFINILM_TRACE_OP(Gemm, attn);
        INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
            && runtime.device().type() != infinilm::runtime::Device::Type::kMars
            && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
        infinilm::execution::submit<infini::ops::Softmax>(
            runtime, infini::ops::Softmax::DefaultConfig(runtime.device().type()), attn, std::int64_t{-1}, std::optional<infini::ops::DataType>{}, attn);
        auto attention_output_shape = attn->shape();
        attention_output_shape.back() = v_flat->size(attn->ndim() - 1);
        auto attention_output = infinilm::runtime::Tensor::empty(runtime, attention_output_shape, attn->dtype());
        infinilm::execution::submit<infini::ops::Gemm>(
            runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
            infini::ops::Gemm::DefaultConfig(runtime.device().type()), attn, v_flat,
            std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0f},
            std::optional<float>{0.0f}, std::optional<int>{}, std::optional<int>{}, attention_output);
        INFINILM_TRACE_OP(Gemm, attention_output);
        return attention_output
            ->view({num_heads_, len, head_dim_})
            ->permute({1, 0, 2})
            ->contiguous(runtime)
            ->view({len, hidden_size_});
    };

    infinilm::runtime::Tensor out;
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
            auto q = qkv_windows->narrow({{2, 0, 1}})->squeeze(2)->permute({0, 2, 1, 3})->contiguous(runtime);
            auto k = qkv_windows->narrow({{2, 1, 1}})->squeeze(2)->permute({0, 2, 1, 3})->contiguous(runtime);
            auto v = qkv_windows->narrow({{2, 2, 1}})->squeeze(2)->permute({0, 2, 1, 3})->contiguous(runtime);
            auto q_flat = q->view({num_windows * num_heads_, first_len, head_dim_});
            auto k_flat = k->view({num_windows * num_heads_, first_len, head_dim_});
            auto v_flat = v->view({num_windows * num_heads_, first_len, head_dim_});
            const auto key_transposed = k_flat->permute({0, 2, 1});
            auto attn_shape = q_flat->shape();
            attn_shape.back() = key_transposed->size(q_flat->ndim() - 1);
            auto attn = infinilm::runtime::Tensor::empty(runtime, attn_shape, q_flat->dtype());
            infinilm::execution::submit<infini::ops::Gemm>(
                runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
                infini::ops::Gemm::DefaultConfig(runtime.device().type()), q_flat, key_transposed,
                std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{scale_},
                std::optional<float>{0.0f}, std::optional<int>{}, std::optional<int>{}, attn);
            INFINILM_TRACE_OP(Gemm, attn);
            INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
                && runtime.device().type() != infinilm::runtime::Device::Type::kMars
                && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
            infinilm::execution::submit<infini::ops::Softmax>(
                runtime, infini::ops::Softmax::DefaultConfig(runtime.device().type()), attn, std::int64_t{-1}, std::optional<infini::ops::DataType>{}, attn);
            auto attention_output_shape = attn->shape();
            attention_output_shape.back() = v_flat->size(attn->ndim() - 1);
            auto attention_output = infinilm::runtime::Tensor::empty(runtime, attention_output_shape, attn->dtype());
            infinilm::execution::submit<infini::ops::Gemm>(
                runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
                infini::ops::Gemm::DefaultConfig(runtime.device().type()), attn, v_flat,
                std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0f},
                std::optional<float>{0.0f}, std::optional<int>{}, std::optional<int>{}, attention_output);
            INFINILM_TRACE_OP(Gemm, attention_output);
            out = attention_output
                      ->view({num_windows, num_heads_, first_len, head_dim_})
                      ->permute({0, 2, 1, 3})
                      ->contiguous(runtime)
                      ->view({seq_len, hidden_size_});
        } else {
            std::vector<infinilm::runtime::Tensor> chunks;
            chunks.reserve(cu_window_seqlens->size() - 1);
            for (size_t i = 0; i + 1 < cu_window_seqlens->size(); ++i) {
                const size_t start = static_cast<size_t>((*cu_window_seqlens)[i]);
                const size_t end = static_cast<size_t>((*cu_window_seqlens)[i + 1]);
                if (end > start) {
                    chunks.push_back(run_dense(start, end - start));
                }
            }
            if (chunks.size() == 1) {
                out = chunks.front();
            } else {
                out = infinilm::execution::cat_alloc(runtime, chunks, 0);
        }
    }
    }
    return proj_->forward(runtime, out);
}

VideoNSAVisionMLP::VideoNSAVisionMLP(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                     const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("VideoNSAVisionMLP construction requires eager Runtime");
    try {
    const size_t hidden_size = config.value("hidden_size", 1280);
    const size_t intermediate_size = config.value("intermediate_size", 3420);
    gate_proj_ = this->bind_child("gate_proj", std::make_shared<std::remove_reference_t<decltype(*gate_proj_ )>>(runtime, hidden_size, intermediate_size, true, dtype));
    up_proj_ = this->bind_child("up_proj", std::make_shared<std::remove_reference_t<decltype(*up_proj_ )>>(runtime, hidden_size, intermediate_size, true, dtype));
    down_proj_ = this->bind_child("down_proj", std::make_shared<std::remove_reference_t<decltype(*down_proj_ )>>(runtime, intermediate_size, hidden_size, true, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor VideoNSAVisionMLP::forward(infinilm::runtime::Runtime &runtime,
                                                const infinilm::runtime::Tensor &hidden_states) const {
    auto hidden = const_cast<infinilm::runtime::Tensor &>(hidden_states);
    auto gate = gate_proj_->forward(runtime, hidden);
    auto up = up_proj_->forward(runtime, hidden);
    auto x = infinilm::runtime::Tensor::empty(runtime, up->shape(), up->dtype());
    infinilm::layers::mlp::branch_activation_into(runtime, x, up, gate);
    return down_proj_->forward(runtime, x);
}

VideoNSAVisionBlock::VideoNSAVisionBlock(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                         const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("VideoNSAVisionBlock construction requires eager Runtime");
    try {
    const size_t hidden_size = config.value("hidden_size", 1280);
    norm1_ = this->bind_child("norm1", std::make_shared<std::remove_reference_t<decltype(*norm1_ )>>(runtime, hidden_size, 1e-6, dtype));
    attn_ = this->bind_child("attn", std::make_shared<std::remove_reference_t<decltype(*attn_ )>>(runtime, config, dtype));
    norm2_ = this->bind_child("norm2", std::make_shared<std::remove_reference_t<decltype(*norm2_ )>>(runtime, hidden_size, 1e-6, dtype));
    mlp_ = this->bind_child("mlp", std::make_shared<std::remove_reference_t<decltype(*mlp_ )>>(runtime, config, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor VideoNSAVisionBlock::forward(infinilm::runtime::Runtime &runtime,
                                                const infinilm::runtime::Tensor &hidden_states,
                                                const std::vector<int64_t> *cu_window_seqlens) const {
    auto residual = hidden_states;
    auto x = norm1_->forward(runtime, hidden_states);
    x = attn_->forward(runtime, x, cu_window_seqlens);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), x, residual, ops_output);
        INFINILM_TRACE_OP(Add, ops_output);
        x = std::move(ops_output);
    }

    residual = x;
    x = norm2_->forward(runtime, x);
    x = mlp_->forward(runtime, x);
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), x, residual, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

VideoNSAPatchMerger::VideoNSAPatchMerger(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                         const infinilm::runtime::DataType &dtype)
    : hidden_size_(config.value("hidden_size", 1280)),
      spatial_merge_unit_(config.value("spatial_merge_size", 2) * config.value("spatial_merge_size", 2)),
      merged_size_(hidden_size_ * spatial_merge_unit_) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("VideoNSAPatchMerger construction requires eager Runtime");
    try {
    const size_t out_hidden_size = config.value("out_hidden_size", 3584);
    ln_q_ = this->bind_child("ln_q", std::make_shared<std::remove_reference_t<decltype(*ln_q_ )>>(runtime, hidden_size_, 1e-6, dtype));
    mlp_0_ = this->bind_child("mlp.0", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, merged_size_, merged_size_, true, dtype));
    mlp_2_ = this->bind_child("mlp.2", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, merged_size_, out_hidden_size, true, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor VideoNSAPatchMerger::forward(infinilm::runtime::Runtime &runtime,
                                                const infinilm::runtime::Tensor &hidden_states) const {
    auto x = ln_q_->forward(runtime, hidden_states)->view({hidden_states->size(0) / spatial_merge_unit_, merged_size_});
    x = mlp_0_->forward(runtime, x);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
        infinilm::execution::submit<infini::ops::Gelu>(
            runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), x, std::string{"none"}, ops_output);
        x = std::move(ops_output);
    }
    return mlp_2_->forward(runtime, x);
}

VideoNSAVisionModel::VideoNSAVisionModel(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                         const infinilm::runtime::DataType &dtype)
    : depth_(config.value("depth", 32)),
      hidden_size_(config.value("hidden_size", 1280)),
      patch_size_(config.value("patch_size", 14)),
      spatial_merge_size_(config.value("spatial_merge_size", 2)),
      spatial_merge_unit_(spatial_merge_size_ * spatial_merge_size_),
      window_size_(config.value("window_size", 112)) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("VideoNSAVisionModel construction requires eager Runtime");
    try {
    if (config.contains("fullatt_block_indexes")) {
        for (const auto &idx : config["fullatt_block_indexes"]) {
            fullatt_block_indexes_.insert(idx.get<size_t>());
        }
    }
    patch_embed_ = this->bind_child("patch_embed", std::make_shared<std::remove_reference_t<decltype(*patch_embed_ )>>(runtime, config, dtype));
    blocks_.reserve(depth_);
    for (size_t i = 0; i < depth_; ++i) {
        blocks_.push_back(this->bind_child("blocks." + std::to_string(i), std::make_shared<VideoNSAVisionBlock>(runtime, config, dtype)));
    }
    merger_ = this->bind_child("merger", std::make_shared<std::remove_reference_t<decltype(*merger_ )>>(runtime, config, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

VideoNSAVisionModel::WindowMetadata VideoNSAVisionModel::build_window_metadata_(infinilm::runtime::Runtime &runtime,
                                                                                    const infinilm::runtime::Tensor &grid_thw) const {
    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    auto grid_cpu = grid_thw->device() == cpu ? grid_thw : grid_thw->to(runtime, cpu);
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

infinilm::runtime::Tensor VideoNSAVisionModel::gather_rows_(infinilm::runtime::Runtime &runtime,
                                                     const infinilm::runtime::Tensor &hidden_states,
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
    auto indices = infinilm::runtime::Tensor::empty(runtime, {rows, width}, infinilm::runtime::DataType::kInt64);
    runtime.memcpyH2D(indices->data(), flat_indices.data(), flat_indices.size() * sizeof(int64_t), false);
    auto contiguous_hidden_states = hidden_states->contiguous(runtime);
    auto gathered_hidden_states = infinilm::runtime::Tensor::empty(runtime, indices->shape(), contiguous_hidden_states->dtype());
    infinilm::execution::tensor_checks::take::validate(runtime, gathered_hidden_states, contiguous_hidden_states, indices);
    infinilm::execution::submit<infini::ops::Take>(runtime, infinilm::execution::RecordingPolicy::DeviceOnly, infini::ops::Take::DefaultConfig(runtime.device().type()), contiguous_hidden_states, indices, gathered_hidden_states);
    return gathered_hidden_states;
}

infinilm::runtime::Tensor VideoNSAVisionModel::forward(infinilm::runtime::Runtime &runtime,
                                                const infinilm::runtime::Tensor &pixel_values,
                                                const infinilm::runtime::Tensor &grid_thw) const {
    auto hidden_states = patch_embed_->forward(runtime, pixel_values);
    auto window_meta = build_window_metadata_(runtime, grid_thw);
    if (window_meta.patch_order.size() != hidden_states->size(0)) {
        throw std::runtime_error("VideoNSAVisionModel: window metadata does not match patch sequence length");
    }
    hidden_states = gather_rows_(runtime, hidden_states, window_meta.patch_order);
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const auto *cu_segments = fullatt_block_indexes_.count(i) > 0 ? &window_meta.cu_full_seqlens : &window_meta.cu_window_seqlens;
        hidden_states = blocks_[i]->forward(runtime, hidden_states, cu_segments);
    }
    auto merged = merger_->forward(runtime, hidden_states);
    return gather_rows_(runtime, merged, window_meta.reverse_order);
}

} // namespace infinilm::models::videonsa
