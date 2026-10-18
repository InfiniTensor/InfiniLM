#include <base/flash_attn_with_kvcache.h>
#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include "../../layers/linear/dense.hpp"
#include "../../global_state/forward_context.hpp"
#include <exception>
#include "runtime/runtime.hpp"
#include "kimi_k3_vision.hpp"

#include <base/upsample_bicubic2d.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace infinilm::models::kimi_k3 {
namespace {

std::vector<int64_t> grid_to_cpu(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &grid_thw) {
    auto cpu = grid_thw->device().type() == infinilm::runtime::Device::Type::kCpu
        ? grid_thw
        : grid_thw->to(runtime, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu, 0});
    std::vector<int64_t> grid(cpu->numel());
    if (cpu->dtype() == infinilm::runtime::DataType::kInt64) {
        const auto *data = reinterpret_cast<const int64_t *>(cpu->data());
        return {data, data + cpu->numel()};
    }
    if (cpu->dtype() == infinilm::runtime::DataType::kInt32) {
        const auto *data = reinterpret_cast<const int32_t *>(cpu->data());
        for (size_t i = 0; i < cpu->numel(); ++i) {
            grid[i] = data[i];
        }
        return grid;
    }
    throw std::runtime_error("KimiK3VisionTower: grid_thw must be int32 or int64");
}

} // namespace

KimiK3VisionPatchProjection::KimiK3VisionPatchProjection(
    infinilm::runtime::Runtime &runtime,
    size_t hidden_size,
                                                         size_t patch_size,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3VisionPatchProjection construction requires eager Runtime");
    try {
        patch_size_ = patch_size;
        weight_ = infinilm::weights::Weight(runtime, {hidden_size, 3, patch_size, patch_size}, dtype);
        this->bind_weight("weight", weight_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK3VisionPatchProjection::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values) const {
    const size_t patch_dim = 3 * patch_size_ * patch_size_;
    auto pixels = pixel_values->view({pixel_values->numel() / patch_dim, patch_dim});
    return infinilm::layers::linear::dense_projection(runtime, pixels, static_cast<infinilm::runtime::Tensor>(weight_->view({weight_->size(0), patch_dim})), std::nullopt, 1.0F, std::nullopt, false);
}

KimiK3VisionPosEmbed::KimiK3VisionPosEmbed(
    infinilm::runtime::Runtime &runtime,
    size_t height,
                                           size_t width,
                                           size_t hidden_size,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3VisionPosEmbed construction requires eager Runtime");
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

infinilm::runtime::Tensor KimiK3VisionPosEmbed::forward(infinilm::runtime::Runtime &runtime, size_t grid_h, size_t grid_w) const {
    if (grid_h == height_ && grid_w == width_) {
        return weight_->view({height_ * width_, hidden_size_});
    }
    auto nchw = weight_->permute({2, 0, 1})->unsqueeze(0);
    runtime.assertOwnerThread();
    if (nchw->device() != runtime.device()) {
        throw std::invalid_argument("UpsampleBicubic requires NCHW input on its Runtime device");
    }
    const std::vector<int64_t> output_size{static_cast<int64_t>(grid_h), static_cast<int64_t>(grid_w)};
    if (output_size[0] <= 0 || output_size[1] <= 0) {
        throw std::invalid_argument("UpsampleBicubic output_size must contain positive H and W");
    }
    auto resized = infinilm::runtime::Tensor::empty(runtime,
        {nchw->size(0), nchw->size(1), static_cast<size_t>(output_size[0]), static_cast<size_t>(output_size[1])},
        nchw->dtype());
    infinilm::execution::submit<infini::ops::UpsampleBicubic2d>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly, infini::ops::Config::Automatic(),
        nchw, output_size, false, std::optional<double>{}, std::optional<double>{}, resized);
    return resized->squeeze(0)
        ->permute({1, 2, 0})
        ->contiguous(runtime)
        ->view({grid_h * grid_w, hidden_size_});
}

KimiK3VisionPatchEmbed::KimiK3VisionPatchEmbed(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3VisionPatchEmbed construction requires eager Runtime");
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

infinilm::runtime::Tensor KimiK3VisionPatchEmbed::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values,
                                                   const infinilm::runtime::Tensor &grid_thw) const {
    const auto grid = grid_to_cpu(runtime, grid_thw);
    if (grid.size() != 3 || grid[0] != 1) {
        // TODO(kimi_k3): Add divided fixed temporal embeddings for video input.
        throw std::runtime_error("KimiK3VisionPatchEmbed: only single-frame image input is supported");
    }
    auto hidden_states = proj_->forward(runtime, pixel_values);
    const auto ops_other = pos_emb_->forward(runtime, static_cast<size_t>(grid[1]), static_cast<size_t>(grid[2]));
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, hidden_states->shape(), hidden_states->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), hidden_states, ops_other, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

KimiK3VisionMLP::KimiK3VisionMLP(
    infinilm::runtime::Runtime &runtime,
    size_t hidden_size,
                                 size_t intermediate_size,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3VisionMLP construction requires eager Runtime");
    try {
        fc0_ = std::make_shared<std::remove_reference_t<decltype(*fc0_)>>(
            runtime, hidden_size, intermediate_size, false, dtype);
        this->bind_child("fc0", fc0_);
        fc1_ = std::make_shared<std::remove_reference_t<decltype(*fc1_)>>(
            runtime, intermediate_size, hidden_size, false, dtype);
        this->bind_child("fc1", fc1_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK3VisionMLP::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    auto mutable_hidden = hidden_states;
    const auto ops_input = fc0_->forward(runtime, mutable_hidden);
    auto intermediate = infinilm::runtime::Tensor::empty(runtime, ops_input->shape(), ops_input->dtype());
    infinilm::execution::submit<infini::ops::Gelu>(
        runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), ops_input, std::string{"tanh"}, intermediate);
    return fc1_->forward(runtime, intermediate);
}

KimiK3VisionBlock::KimiK3VisionBlock(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3VisionBlock construction requires eager Runtime");
    try {
        hidden_size_ = config.at("vt_hidden_size").get<size_t>();
        num_heads_ = config.at("vt_num_attention_heads").get<size_t>();
        if (!hidden_size_ || !num_heads_ || !config.at("qkv_hidden_size").get<size_t>() || config.at("qkv_hidden_size").get<size_t>() % num_heads_ != 0)
            throw std::invalid_argument("Invalid KimiK3VisionBlock head geometry");
        head_dim_ = config.at("qkv_hidden_size").get<size_t>() / num_heads_;
        scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        rotary_emb_ = std::move(rotary_emb);
        norm0_ = std::make_shared<std::remove_reference_t<decltype(*norm0_)>>(
            runtime, hidden_size_, 1e-5, dtype);
        this->bind_child("norm0", norm0_);
        norm1_ = std::make_shared<std::remove_reference_t<decltype(*norm1_)>>(
            runtime, hidden_size_, 1e-5, dtype);
        this->bind_child("norm1", norm1_);
    const size_t qkv_hidden_size = config.at("qkv_hidden_size").get<size_t>();
        wqkv_ = std::make_shared<std::remove_reference_t<decltype(*wqkv_)>>(
            runtime, hidden_size_, qkv_hidden_size * 3, false, dtype);
        this->bind_child("wqkv", wqkv_);
        wo_ = std::make_shared<std::remove_reference_t<decltype(*wo_)>>(
            runtime, qkv_hidden_size, hidden_size_, false, dtype);
        this->bind_child("wo", wo_);
        mlp_ = std::make_shared<std::remove_reference_t<decltype(*mlp_)>>(
            runtime, hidden_size_, config.at("vt_intermediate_size").get<size_t>(), dtype);
        this->bind_child("mlp", mlp_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK3VisionBlock::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
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
        const auto mha_config = infini::ops::FlashAttnWithKvcache::DefaultConfig(runtime.device().type());
        const std::optional<infinilm::runtime::Tensor> mha_none;
        infinilm::execution::submit<infini::ops::FlashAttnWithKvcache>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, mha_config,
            mha_query, mha_key, mha_value, mha_none, mha_none, mha_none, mha_none,
            mha_none, mha_none, mha_none, mha_none, mha_none, std::optional<double>{scale_},
            false, std::vector<int64_t>{-1, -1}, double{0.0}, true, int64_t{0}, false,
            mha_output, mha_none);
        INFINILM_TRACE_OP(MultiheadAttention, mha_output);
    }
    auto attn_output = mha_output->view({seq_len, num_heads_ * head_dim_});
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

KimiK3VisionEncoder::KimiK3VisionEncoder(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3VisionEncoder construction requires eager Runtime");
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb;
    std::shared_ptr<KimiK3VisionBlock> pending_child;
    try {
    const size_t hidden_size = config.at("vt_hidden_size").get<size_t>();
    const size_t num_heads = config.at("vt_num_attention_heads").get<size_t>();
        if (!hidden_size || !num_heads || !config.at("qkv_hidden_size").get<size_t>() || config.at("qkv_hidden_size").get<size_t>() % num_heads != 0)
            throw std::invalid_argument("Invalid KimiK3VisionEncoder head geometry");
    const size_t head_dim = config.at("qkv_hidden_size").get<size_t>() / num_heads;
        rotary_emb = std::make_shared<infinilm::layers::RoPE>(runtime,
        head_dim / 2, head_dim / 2, 512, 10000.0,
            infinilm::layers::RoPE::Algo::GPT_J, dtype, runtime.device());
    const size_t num_layers = config.at("vt_num_hidden_layers").get<size_t>();
    blocks_.reserve(num_layers);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            pending_child = std::make_shared<KimiK3VisionBlock>(
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

infinilm::runtime::Tensor KimiK3VisionEncoder::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                                                const infinilm::runtime::Tensor &grid_thw) const {
    const auto grid = grid_to_cpu(runtime, grid_thw);
    if (grid.size() != 3 || grid[0] != 1) {
        throw std::runtime_error("KimiK3VisionEncoder: only one image grid is supported per call");
    }
    const size_t grid_h = static_cast<size_t>(grid[1]);
    const size_t grid_w = static_cast<size_t>(grid[2]);
    auto positions_storage = std::make_shared<std::vector<int64_t>>(2 * grid_h * grid_w);
    auto positions_cpu = infinilm::runtime::Tensor::from_blob(
        positions_storage->data(), {2, grid_h * grid_w}, infinilm::runtime::DataType::kInt64,
        infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu, 0}, positions_storage);
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

KimiK3VisionTower::KimiK3VisionTower(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3VisionTower construction requires eager Runtime");
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

infinilm::runtime::Tensor KimiK3VisionTower::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values,
                                              const infinilm::runtime::Tensor &grid_thw) const {
    const auto grid = grid_to_cpu(runtime, grid_thw);
    const size_t grid_h = static_cast<size_t>(grid.at(1));
    const size_t grid_w = static_cast<size_t>(grid.at(2));
    if (grid.at(0) != 1 || grid_h % merge_height_ != 0 || grid_w % merge_width_ != 0) {
        // TODO(kimi_k3): Implement temporal pooling for video grids with t > 1.
        throw std::runtime_error("KimiK3VisionTower: unsupported image/video merge grid");
    }
    auto hidden_states = encoder_->forward(runtime, patch_embed_->forward(runtime, pixel_values, grid_thw), grid_thw);
    return hidden_states
        ->view({grid_h / merge_height_, merge_height_, grid_w / merge_width_, merge_width_, hidden_size_})
        ->permute({0, 2, 1, 3, 4})
        ->contiguous(runtime)
        ->view({grid_h * grid_w / (merge_height_ * merge_width_), merge_height_ * merge_width_, hidden_size_});
}

KimiK3Projector::KimiK3Projector(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &config,
    const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("KimiK3Projector construction requires eager Runtime");
    try {
    const size_t vision_hidden_size = config.at("mm_hidden_size").get<size_t>();
    const auto &kernel = config.at("merge_kernel_size");
    merged_hidden_size_ = vision_hidden_size * kernel.at(0).get<size_t>() * kernel.at(1).get<size_t>();
    const size_t text_hidden_size = config.at("text_hidden_size").get<size_t>();
    const double eps = config.at("projector_ln_eps").get<double>();
        linear_0_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, merged_hidden_size_, merged_hidden_size_, false, dtype);
        this->bind_child("proj.0", linear_0_);
        linear_2_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, merged_hidden_size_, text_hidden_size, false, dtype);
        this->bind_child("proj.2", linear_2_);
        post_norm_ = std::make_shared<std::remove_reference_t<decltype(*post_norm_)>>(
            runtime, text_hidden_size, eps, dtype);
        this->bind_child("post_norm", post_norm_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor KimiK3Projector::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &vision_features) const {
    auto flattened = vision_features->view({vision_features->size(0), merged_hidden_size_});
    const auto ops_input = linear_0_->forward(runtime, flattened);
    auto hidden = infinilm::runtime::Tensor::empty(runtime, ops_input->shape(), ops_input->dtype());
    infinilm::execution::submit<infini::ops::Gelu>(
        runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), ops_input, std::string{"none"}, hidden);
    return post_norm_->forward(runtime, linear_2_->forward(runtime, hidden));
}

} // namespace infinilm::models::kimi_k3
