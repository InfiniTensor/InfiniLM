#ifndef INFINICORE_MHA_OMITTED
#include <base/flash_attn_with_kvcache.h>
#endif
#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/linear/dense.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include "../../global_state/forward_context.hpp"
#include <exception>
#include "kimi_k25_vision.hpp"

#if !defined(INFINICORE_UPSAMPLE_BICUBIC_OPS_UNAVAILABLE)
#include <base/upsample_bicubic2d.h>
#endif

#include "runtime/runtime.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace infinilm::models::kimi_k25 {
namespace {

std::vector<int64_t> grid_to_cpu(infinilm::runtime::Runtime &runtime,
                                 const infinilm::runtime::Tensor &grid_thw) {
    runtime.assertOwnerThread();
    if (!grid_thw || grid_thw->numel() != 3 ||
        (grid_thw->dtype() != infinilm::runtime::DataType::kInt32 &&
         grid_thw->dtype() != infinilm::runtime::DataType::kInt64)) {
        throw std::runtime_error("KimiK25VisionTower: grid_thw must contain three int32 or int64 values");
    }
    const auto cpu_device = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    std::array<std::byte, 3 * sizeof(int64_t)> device_bytes{};
    const std::byte *data = nullptr;
    const bool on_cpu = grid_thw->device() == cpu_device;
    if (on_cpu) {
        data = grid_thw->data();
    } else {
        if (grid_thw->device() != runtime.device()) {
            throw std::runtime_error("KimiK25VisionTower: grid_thw device differs from Runtime");
        }
        auto contiguous = grid_thw->contiguous(runtime);
        runtime.memcpyD2H(device_bytes.data(), contiguous->data(), contiguous->nbytes());
        data = device_bytes.data();
    }
    std::vector<int64_t> values(3);
    for (size_t linear = 0; linear < 3; ++linear) {
        infinilm::runtime::Stride offset = 0;
        if (on_cpu) {
            size_t remaining = linear;
            for (size_t axis = grid_thw->ndim(); axis > 0; --axis) {
                const size_t dim = grid_thw->size(axis - 1);
                offset += static_cast<infinilm::runtime::Stride>(remaining % dim) * grid_thw->stride(axis - 1);
                remaining /= dim;
            }
        } else {
            offset = static_cast<infinilm::runtime::Stride>(linear);
        }
        if (grid_thw->dtype() == infinilm::runtime::DataType::kInt64) {
            int64_t value;
            std::memcpy(&value, data + offset * static_cast<infinilm::runtime::Stride>(sizeof(value)), sizeof(value));
            values[linear] = value;
        } else {
            int32_t value;
            std::memcpy(&value, data + offset * static_cast<infinilm::runtime::Stride>(sizeof(value)), sizeof(value));
            values[linear] = value;
        }
    }
    return values;
}

} // namespace

KimiK25VisionPatchProjection::KimiK25VisionPatchProjection(
    infinilm::runtime::Runtime &runtime,
    size_t hidden_size,
                                                           size_t patch_size,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK25VisionPatchProjection construction requires eager Runtime");
    try {
        patch_size_ = patch_size;
        weight_ = infinilm::weights::Weight(runtime, {hidden_size, 3, patch_size, patch_size}, dtype);
        this->bind_weight("weight", weight_);
        bias_ = infinilm::weights::Weight(runtime, {hidden_size}, dtype);
        this->bind_weight("bias", bias_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK25VisionPatchProjection::forward(infinilm::runtime::Runtime &runtime,
                                                          const infinilm::runtime::Tensor &pixel_values) const {
    runtime.assertOwnerThread();
    if (!pixel_values || pixel_values->device() != runtime.device()) {
        throw std::runtime_error("KimiK25VisionPatchProjection: pixel_values device differs from Runtime");
    }
    const size_t patch_dim = 3 * patch_size_ * patch_size_;
    auto pixels = pixel_values->view({pixel_values->numel() / patch_dim, patch_dim});
    return infinilm::layers::linear::dense_projection(runtime, pixels, weight_->view({weight_->size(0), patch_dim}), std::make_optional<infinilm::runtime::Tensor>(bias_.tensor()));
}

KimiK25VisionPosEmbed::KimiK25VisionPosEmbed(
    infinilm::runtime::Runtime &runtime,
    size_t height,
                                             size_t width,
                                             size_t hidden_size,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK25VisionPosEmbed construction requires eager Runtime");
    try {
        height_ = height;
        width_ = width;
        hidden_size_ = hidden_size;
        weight_ = infinilm::weights::Weight(runtime, {height_, width_, hidden_size_}, dtype);
        this->bind_weight("weight", weight_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK25VisionPosEmbed::forward(infinilm::runtime::Runtime &runtime,
                                                  size_t grid_h, size_t grid_w) const {
    runtime.assertOwnerThread();
    if (weight_->device() != runtime.device()) {
        throw std::runtime_error("KimiK25VisionPosEmbed: weight device differs from Runtime");
    }
    if (!grid_h || !grid_w ||
        grid_h > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        grid_w > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("KimiK25VisionPosEmbed: invalid grid dimensions");
    }
    if (grid_h == height_ && grid_w == width_) {
        return weight_->view({height_ * width_, hidden_size_});
    }
#if !defined(INFINICORE_UPSAMPLE_BICUBIC_OPS_UNAVAILABLE)
    auto nchw = weight_->permute({2, 0, 1})->unsqueeze(0);
    auto resized = infinilm::runtime::Tensor::empty(
        runtime, {nchw->size(0), nchw->size(1), grid_h, grid_w}, nchw->dtype());
    const std::vector<int64_t> output_size{static_cast<int64_t>(grid_h), static_cast<int64_t>(grid_w)};
    infinilm::execution::submit<infini::ops::UpsampleBicubic2d>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::UpsampleBicubic2d::DefaultConfig(runtime.device().type()),
        nchw, output_size, false, std::optional<double>{}, std::optional<double>{}, resized);
    return resized->squeeze(0)
        ->permute({1, 2, 0})
        ->contiguous(runtime)
        ->view({grid_h * grid_w, hidden_size_});
#else
    throw std::runtime_error("UpsampleBicubic requires an InfiniOps prefix with UpsampleBicubic2d");
#endif
}

KimiK25VisionPatchEmbed::KimiK25VisionPatchEmbed(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK25VisionPatchEmbed construction requires eager Runtime");
    try {
    const size_t hidden_size = config.at("vt_hidden_size").get<size_t>();
        proj_ = std::make_shared<std::remove_reference_t<decltype(*proj_)>>(
            runtime, hidden_size, config.at("patch_size").get<size_t>(), dtype);
        this->bind_child("proj", proj_);
        pos_emb_ = std::make_shared<std::remove_reference_t<decltype(*pos_emb_)>>(
            runtime, config.at("init_pos_emb_height").get<size_t>(), config.at("init_pos_emb_width").get<size_t>(), hidden_size, dtype);
        this->bind_child("pos_emb", pos_emb_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK25VisionPatchEmbed::forward(infinilm::runtime::Runtime &runtime,
                                                    const infinilm::runtime::Tensor &pixel_values,
                                                    const infinilm::runtime::Tensor &grid_thw) const {
    const auto grid = grid_to_cpu(runtime, grid_thw);
    if (grid[0] != 1) {
        // TODO(kimi_k25): Add the fixed temporal positional embedding for video inputs.
        throw std::runtime_error("KimiK25VisionPatchEmbed: only single-frame image input is supported");
    }
    if (grid[1] <= 0 || grid[2] <= 0) {
        throw std::runtime_error("KimiK25VisionPatchEmbed: grid dimensions must be positive");
    }
    auto hidden_states = proj_->forward(runtime, pixel_values);
    const auto ops_other = pos_emb_->forward(runtime, static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2]));
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, hidden_states->shape(), hidden_states->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), hidden_states, ops_other, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

KimiK25VisionMLP::KimiK25VisionMLP(
    infinilm::runtime::Runtime &runtime,
    size_t hidden_size,
                                   size_t intermediate_size,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK25VisionMLP construction requires eager Runtime");
    try {
        fc0_ = std::make_shared<std::remove_reference_t<decltype(*fc0_)>>(
            runtime, hidden_size, intermediate_size, true, dtype);
        this->bind_child("fc0", fc0_);
        fc1_ = std::make_shared<std::remove_reference_t<decltype(*fc1_)>>(
            runtime, intermediate_size, hidden_size, true, dtype);
        this->bind_child("fc1", fc1_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK25VisionMLP::forward(infinilm::runtime::Runtime &runtime,
                                             const infinilm::runtime::Tensor &hidden_states) const {
    auto mutable_hidden = hidden_states;
    const auto ops_input = fc0_->forward(runtime, mutable_hidden);
    auto intermediate = infinilm::runtime::Tensor::empty(runtime, ops_input->shape(), ops_input->dtype());
    infinilm::execution::submit<infini::ops::Gelu>(
        runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), ops_input, std::string{"tanh"}, intermediate);
    return fc1_->forward(runtime, intermediate);
}

KimiK25VisionBlock::KimiK25VisionBlock(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK25VisionBlock construction requires eager Runtime");
    try {
        hidden_size_ = config.at("vt_hidden_size").get<size_t>();
        num_heads_ = config.at("vt_num_attention_heads").get<size_t>();
        if (!hidden_size_ || !num_heads_ || !hidden_size_ || hidden_size_ % num_heads_ != 0)
            throw std::invalid_argument("Invalid KimiK25VisionBlock head geometry");
        head_dim_ = hidden_size_ / num_heads_;
        scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        rotary_emb_ = std::move(rotary_emb);
        norm0_ = std::make_shared<std::remove_reference_t<decltype(*norm0_)>>(
            runtime, hidden_size_, 1e-5, dtype);
        this->bind_child("norm0", norm0_);
        norm1_ = std::make_shared<std::remove_reference_t<decltype(*norm1_)>>(
            runtime, hidden_size_, 1e-5, dtype);
        this->bind_child("norm1", norm1_);
        wqkv_ = std::make_shared<std::remove_reference_t<decltype(*wqkv_)>>(
            runtime, hidden_size_, hidden_size_ * 3, true, dtype);
        this->bind_child("wqkv", wqkv_);
        wo_ = std::make_shared<std::remove_reference_t<decltype(*wo_)>>(
            runtime, hidden_size_, hidden_size_, true, dtype);
        this->bind_child("wo", wo_);
        mlp_ = std::make_shared<std::remove_reference_t<decltype(*mlp_)>>(
            runtime, hidden_size_, config.at("vt_intermediate_size").get<size_t>(), dtype);
        this->bind_child("mlp", mlp_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK25VisionBlock::forward(infinilm::runtime::Runtime &runtime,
                                               const infinilm::runtime::Tensor &hidden_states,
                                               const infinilm::runtime::Tensor &row_positions,
                                               const infinilm::runtime::Tensor &col_positions) const {
    const size_t seq_len = hidden_states->size(0);
    auto normalized = norm0_->forward(runtime, hidden_states);
    auto qkv = wqkv_->forward(runtime, normalized)->view({seq_len, 3, num_heads_, head_dim_});
    auto q = qkv->narrow({{1, 0, 1}})->squeeze(1)->contiguous(runtime);
    auto k = qkv->narrow({{1, 1, 1}})->squeeze(1)->contiguous(runtime);
    auto v = qkv->narrow({{1, 2, 1}})->squeeze(1)->contiguous(runtime);

    const size_t axis_dim = head_dim_ / 2;
    const size_t axis_pairs = axis_dim / 2;
    auto apply_2d_rope = [&](const infinilm::runtime::Tensor &x) {
        auto col = infinilm::runtime::Tensor::empty(runtime, {seq_len, num_heads_, axis_dim}, x->dtype());
        auto row = infinilm::runtime::Tensor::empty(runtime, {seq_len, num_heads_, axis_dim}, x->dtype());
        for (size_t pair = 0; pair < axis_pairs; ++pair) {
            col->narrow({{2, pair * 2, 2}})->copy_from(runtime, x->narrow({{2, pair * 4, 2}}));
            row->narrow({{2, pair * 2, 2}})->copy_from(runtime, x->narrow({{2, pair * 4 + 2, 2}}));
        }
        rotary_emb_->forward(runtime, col, col_positions, true);
        rotary_emb_->forward(runtime, row, row_positions, true);
        for (size_t pair = 0; pair < axis_pairs; ++pair) {
            x->narrow({{2, pair * 4, 2}})->copy_from(runtime, col->narrow({{2, pair * 2, 2}}));
            x->narrow({{2, pair * 4 + 2, 2}})->copy_from(runtime, row->narrow({{2, pair * 2, 2}}));
        }
    };
    apply_2d_rope(q);
    apply_2d_rope(k);

    const infinilm::runtime::Tensor mha_query = q->unsqueeze(0);
    const infinilm::runtime::Tensor mha_key = k->unsqueeze(0);
    const infinilm::runtime::Tensor mha_value = v->unsqueeze(0);
    auto mha_output = infinilm::runtime::Tensor::empty(runtime, mha_query->shape(), mha_query->dtype());
    {
#ifdef INFINICORE_MHA_OMITTED
        throw std::runtime_error("ordinary MHA is unavailable in the greedy_bootstrap or InfiniOps-disabled build profile");
#else
        const auto mha_config = infini::ops::FlashAttnWithKvcache::DefaultConfig(runtime.device().type());
        const std::optional<infinilm::runtime::Tensor> mha_none;
        infinilm::execution::submit<infini::ops::FlashAttnWithKvcache>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, mha_config,
            mha_query, mha_key, mha_value, mha_none, mha_none, mha_none, mha_none,
            mha_none, mha_none, mha_none, mha_none, mha_none, std::optional<double>{scale_},
            false, std::vector<int64_t>{-1, -1}, double{0.0}, true, int64_t{0}, false,
            mha_output, mha_none);
        INFINILM_TRACE_OP(MultiheadAttention, mha_output);
#endif
    }
    auto attn_output = mha_output->view({seq_len, hidden_size_});
    const auto ops_other = wo_->forward(runtime, attn_output);
    auto attention_residual = infinilm::runtime::Tensor::empty(runtime, hidden_states->shape(), hidden_states->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), hidden_states, ops_other, attention_residual);
    INFINILM_TRACE_OP(Add, attention_residual);
    auto mlp_input = norm1_->forward(runtime, attention_residual);
    const auto ops_mlp_output = mlp_->forward(runtime, mlp_input);
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, attention_residual->shape(), attention_residual->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), attention_residual, ops_mlp_output, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

KimiK25VisionEncoder::KimiK25VisionEncoder(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK25VisionEncoder construction requires eager Runtime");
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb;
    std::shared_ptr<KimiK25VisionBlock> pending_child;
    try {
    const size_t hidden_size = config.at("vt_hidden_size").get<size_t>();
    const size_t num_heads = config.at("vt_num_attention_heads").get<size_t>();
        if (!hidden_size || !num_heads || !hidden_size || hidden_size % num_heads != 0)
            throw std::invalid_argument("Invalid KimiK25VisionEncoder head geometry");
    const size_t head_dim = hidden_size / num_heads;
        rotary_emb = std::make_shared<infinilm::layers::RoPE>(runtime,
        head_dim / 2, head_dim / 2, 512, 10000.0,
            infinilm::layers::RoPE::Algo::GPT_J, dtype, runtime.device());
    const size_t num_layers = config.at("vt_num_hidden_layers").get<size_t>();
    blocks_.reserve(num_layers);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            pending_child = std::make_shared<KimiK25VisionBlock>(
                runtime, config, rotary_emb, dtype);
            blocks_.push_back(pending_child);
            this->bind_child("blocks." + std::to_string(layer_idx), pending_child);
    }
        final_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*final_layernorm_)>>(
            runtime, hidden_size, 1e-5, dtype);
        this->bind_child("final_layernorm", final_layernorm_);
        runtime.syncStream(); // Local RoPE and children remain alive until completion.
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK25VisionEncoder::forward(infinilm::runtime::Runtime &runtime,
                                                 const infinilm::runtime::Tensor &hidden_states,
                                                 const infinilm::runtime::Tensor &grid_thw) const {
    const auto grid = grid_to_cpu(runtime, grid_thw);
    if (grid.size() != 3 || grid[0] != 1) {
        throw std::runtime_error("KimiK25VisionEncoder: only one image grid is supported per call");
    }
    const size_t grid_h = static_cast<size_t>(grid[1]);
    const size_t grid_w = static_cast<size_t>(grid[2]);
    auto positions_storage = std::make_shared<std::vector<int64_t>>(2 * grid_h * grid_w);
    auto positions_cpu = infinilm::runtime::Tensor::from_blob(
        positions_storage->data(), {2, grid_h * grid_w}, infinilm::runtime::DataType::kInt64,
        infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu}, positions_storage);
    auto *positions = positions_storage->data();
    for (size_t row = 0; row < grid_h; ++row) {
        for (size_t col = 0; col < grid_w; ++col) {
            const size_t token = row * grid_w + col;
            positions[token] = static_cast<int64_t>(row);
            positions[grid_h * grid_w + token] = static_cast<int64_t>(col);
        }
    }
    auto positions_device = infinilm::runtime::Tensor::empty(
        runtime, {2, grid_h * grid_w}, infinilm::runtime::DataType::kInt64);
    positions_device->copy_from(runtime, positions_cpu);
    auto row_positions = positions_device->narrow({{0, 0, 1}})->view({grid_h * grid_w});
    auto col_positions = positions_device->narrow({{0, 1, 1}})->view({grid_h * grid_w});

    auto output = hidden_states;
    for (const auto &block : blocks_) {
        output = block->forward(runtime, output, row_positions, col_positions);
    }
    return final_layernorm_->forward(runtime, output);
}

KimiK25VisionTower::KimiK25VisionTower(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK25VisionTower construction requires eager Runtime");
    try {
        hidden_size_ = config.at("vt_hidden_size").get<size_t>();
    const auto &kernel = config.at("merge_kernel_size");
    merge_height_ = kernel.at(0).get<size_t>();
    merge_width_ = kernel.at(1).get<size_t>();
        patch_embed_ = std::make_shared<std::remove_reference_t<decltype(*patch_embed_)>>(
            runtime, config, dtype);
        this->bind_child("patch_embed", patch_embed_);
        encoder_ = std::make_shared<std::remove_reference_t<decltype(*encoder_)>>(
            runtime, config, dtype);
        this->bind_child("encoder", encoder_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK25VisionTower::forward(infinilm::runtime::Runtime &runtime,
                                               const infinilm::runtime::Tensor &pixel_values,
                                               const infinilm::runtime::Tensor &grid_thw) const {
    const auto grid = grid_to_cpu(runtime, grid_thw);
    if (grid[1] <= 0 || grid[2] <= 0) {
        throw std::runtime_error("KimiK25VisionTower: grid dimensions must be positive");
    }
    const size_t grid_h = static_cast<size_t>(grid[1]);
    const size_t grid_w = static_cast<size_t>(grid[2]);
    if (grid.at(0) != 1 || grid_h % merge_height_ != 0 || grid_w % merge_width_ != 0) {
        // TODO(kimi_k25): Implement temporal pooling for video grids with t > 1.
        throw std::runtime_error("KimiK25VisionTower: unsupported image/video merge grid");
    }
    auto hidden_states = encoder_->forward(runtime, patch_embed_->forward(runtime, pixel_values, grid_thw), grid_thw);
    return hidden_states
        ->view({grid_h / merge_height_, merge_height_, grid_w / merge_width_, merge_width_, hidden_size_})
        ->permute({0, 2, 1, 3, 4})
        ->contiguous(runtime)
        ->view({grid_h * grid_w / (merge_height_ * merge_width_), merge_height_ * merge_width_, hidden_size_});
}

KimiK25Projector::KimiK25Projector(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK25Projector construction requires eager Runtime");
    try {
    const size_t vision_hidden_size = config.at("mm_hidden_size").get<size_t>();
    const auto &kernel = config.at("merge_kernel_size");
    merged_hidden_size_ = vision_hidden_size * kernel.at(0).get<size_t>() * kernel.at(1).get<size_t>();
    const size_t text_hidden_size = config.at("text_hidden_size").get<size_t>();
    const double eps = config.at("projector_ln_eps").get<double>();
        pre_norm_ = std::make_shared<std::remove_reference_t<decltype(*pre_norm_)>>(
            runtime, vision_hidden_size, eps, dtype);
        this->bind_child("pre_norm", pre_norm_);
        linear_0_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, merged_hidden_size_, merged_hidden_size_, true, dtype);
        this->bind_child("proj.0", linear_0_);
        linear_2_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, merged_hidden_size_, text_hidden_size, true, dtype);
        this->bind_child("proj.2", linear_2_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK25Projector::forward(infinilm::runtime::Runtime &runtime,
                                             const infinilm::runtime::Tensor &vision_features) const {
    auto normalized = pre_norm_->forward(runtime, vision_features)
                          ->view({vision_features->size(0), merged_hidden_size_});
    const auto ops_input = linear_0_->forward(runtime, normalized);
    auto hidden = infinilm::runtime::Tensor::empty(runtime, ops_input->shape(), ops_input->dtype());
    infinilm::execution::submit<infini::ops::Gelu>(
        runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), ops_input, std::string{"none"}, hidden);
    return linear_2_->forward(runtime, hidden);
}

} // namespace infinilm::models::kimi_k25
