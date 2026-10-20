#include "ernie4_5_vision.hpp"
#include "../../execution/submit.hpp"
#include "../../execution/tensor_checks.hpp"
#include "../../layers/normalization/checks.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include "../../weights/weights.hpp"
#include "runtime/runtime.hpp"
#include <base/flash_attn_with_kvcache.h>
#include <base/quick_gelu.h>
#include <exception>
#include <limits>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::ernie4_5_vl {
namespace {

std::vector<int64_t> tensor_to_i64_vector(infinilm::runtime::Runtime &runtime,
                                          const infinilm::runtime::Tensor &tensor) {
    auto cpu_tensor = tensor->device().type() == infinilm::runtime::Device::Type::kCpu
                        ? tensor
                        : tensor->to(runtime, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    std::vector<int64_t> values(cpu_tensor->numel());
    if (cpu_tensor->dtype() == infinilm::runtime::DataType::kInt64) {
        const auto *ptr = reinterpret_cast<const int64_t *>(cpu_tensor->data());
        values.assign(ptr, ptr + cpu_tensor->numel());
        return values;
    }
    if (cpu_tensor->dtype() == infinilm::runtime::DataType::kInt32) {
        const auto *ptr = reinterpret_cast<const int32_t *>(cpu_tensor->data());
        for (size_t i = 0; i < cpu_tensor->numel(); ++i) {
            values[i] = static_cast<int64_t>(ptr[i]);
        }
        return values;
    }
    throw std::runtime_error("ERNIE 4.5 VL grid_thw must be int32 or int64");
}

infinilm::runtime::Tensor cat_or_single(infinilm::runtime::Runtime &runtime,
                                        std::vector<infinilm::runtime::Tensor> tensors, int dim) {
    if (tensors.empty()) {
        throw std::runtime_error("ERNIE 4.5 VL internal cat received no tensors");
    }
    if (tensors.size() == 1) {
        return tensors[0];
    }
    auto inputs = tensors;
    const auto layout = infinilm::execution::tensor_checks::cat::analyze(runtime, inputs, dim);
    auto output_shape = (*layout.reference)->shape();
    output_shape[layout.axis] = layout.length;
    auto output = infinilm::runtime::Tensor::empty(runtime, output_shape, (*layout.reference)->dtype());
    infinilm::execution::tensor_checks::cat::validate(runtime, output, inputs, layout);
    inputs.erase(std::remove_if(inputs.begin(), inputs.end(), infinilm::execution::tensor_checks::cat::empty_vector), inputs.end());
    if (output->numel()) {
        infinilm::execution::submit<infini::ops::Cat>(runtime, infini::ops::Cat::DefaultConfig(runtime.device().type()), inputs, static_cast<int64_t>(layout.axis), output);
    }
    return output;
}

} // namespace

Ernie45VisionLayerNorm::Ernie45VisionLayerNorm(infinilm::runtime::Runtime &runtime,
                                               size_t normalized_shape,
                                               double eps,
                                               const infinilm::runtime::DataType &dtype)
    : eps_(eps) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Ernie45 vision construction requires eager Runtime");
    }
    try {
        weight_ = infinilm::weights::Weight(runtime, {normalized_shape}, dtype);
        this->bind_weight("weight", weight_);
        bias_ = infinilm::weights::Weight(runtime, {normalized_shape}, dtype);
        this->bind_weight("bias", bias_);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Ernie45VisionLayerNorm::forward(infinilm::runtime::Runtime &runtime,
                                                          const infinilm::runtime::Tensor &hidden_states) const {
    infinilm::layers::check_layer_norm(runtime, hidden_states, weight_.tensor(), bias_.tensor());
    const auto &bias = bias_.tensor();
    const std::optional<infinilm::runtime::Tensor> ops_bias = bias
                                                                ? std::optional<infinilm::runtime::Tensor>{bias}
                                                                : std::nullopt;
    const float epsilon = static_cast<float>(eps_);
    auto y = infinilm::runtime::Tensor::empty(runtime, hidden_states->shape(), hidden_states->dtype());
    infinilm::execution::submit<infini::ops::LayerNorm>(runtime,
                                                        infini::ops::LayerNorm::DefaultConfig(runtime.device().type()),
                                                        hidden_states, weight_.tensor(), ops_bias, epsilon, y);
    INFINILM_TRACE_OP(LayerNorm, y);
    return y;
}

Ernie45VisionPatchEmbed::Ernie45VisionPatchEmbed(infinilm::runtime::Runtime &runtime,
                                                 const nlohmann::json &config,
                                                 const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Ernie45 vision construction requires eager Runtime");
    }
    try {
        const size_t in_channels = config.value("in_channels", config.value("in_chans", 3));
        const size_t patch_size = config.value("patch_size", config.value("spatial_patch_size", 14));
        const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
        proj_ = std::make_shared<std::remove_reference_t<decltype(*proj_)>>(runtime, in_channels * patch_size * patch_size, embed_dim, false, dtype);
        this->bind_child("proj", proj_);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Ernie45VisionPatchEmbed::forward(infinilm::runtime::Runtime &runtime,
                                                           const infinilm::runtime::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    return proj_->forward(runtime, hidden_states_mutable);
}

Ernie45VisionAttention::Ernie45VisionAttention(infinilm::runtime::Runtime &runtime,
                                               const nlohmann::json &config,
                                               const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Ernie45 vision construction requires eager Runtime");
    }
    try {
        hidden_size_ = config.value("embed_dim", config.value("hidden_size", 1280));
        num_heads_ = config.value("num_heads", 16);
        if (hidden_size_ == 0 || num_heads_ == 0) {
            throw std::invalid_argument("Ernie45VisionAttention requires positive hidden_size and num_heads");
        }
        if (hidden_size_ % num_heads_ != 0) {
            throw std::runtime_error("Ernie45VisionAttention: hidden_size must be divisible by num_heads");
        }
        head_dim_ = hidden_size_ / num_heads_;
        if (head_dim_ % 4 != 0) {
            throw std::runtime_error("Ernie45VisionAttention: head_dim must be divisible by 4 for 2D RoPE");
        }
        scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        const size_t axis_head_dim = head_dim_ / 2;
        rotary_emb_ = std::make_shared<std::remove_reference_t<decltype(*rotary_emb_)>>(runtime, axis_head_dim, axis_head_dim, 8192, 10000.0, infinilm::layers::RoPE::Algo::GPT_NEOX, dtype, runtime.device());
        this->bind_child("rotary_emb", rotary_emb_);
        qkv_ = std::make_shared<std::remove_reference_t<decltype(*qkv_)>>(runtime, hidden_size_, hidden_size_ * 3, true, dtype);
        this->bind_child("qkv", qkv_);
        proj_ = std::make_shared<std::remove_reference_t<decltype(*proj_)>>(runtime, hidden_size_, hidden_size_, true, dtype);
        this->bind_child("proj", proj_);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Ernie45VisionAttention::forward(infinilm::runtime::Runtime &runtime,
                                                          const infinilm::runtime::Tensor &hidden_states,
                                                          const infinilm::runtime::Tensor &row_position_ids,
                                                          const infinilm::runtime::Tensor &col_position_ids) const {
    const size_t seq_len = hidden_states->size(0);
    const size_t axis_head_dim = head_dim_ / 2;
    const size_t axis_pair_dim = axis_head_dim / 2;

    auto hidden_mut = hidden_states;
    auto qkv = qkv_->forward(runtime, hidden_mut)->view({seq_len, 3, num_heads_, head_dim_});
    auto q = qkv->narrow({{1, 0, 1}})->squeeze(1)->contiguous(runtime);
    auto k = qkv->narrow({{1, 1, 1}})->squeeze(1)->contiguous(runtime);
    auto v = qkv->narrow({{1, 2, 1}})->squeeze(1)->contiguous(runtime)->unsqueeze(0);

    auto apply_2d_rope = [&](const infinilm::runtime::Tensor &x) {
        auto row = infinilm::runtime::Tensor::empty(runtime, {seq_len, num_heads_, axis_head_dim}, x->dtype());
        auto col = infinilm::runtime::Tensor::empty(runtime, {seq_len, num_heads_, axis_head_dim}, x->dtype());
        row->narrow({{2, 0, axis_pair_dim}})->copy_from(runtime, x->narrow({{2, 0, axis_pair_dim}}));
        row->narrow({{2, axis_pair_dim, axis_pair_dim}})->copy_from(runtime, x->narrow({{2, axis_head_dim, axis_pair_dim}}));
        col->narrow({{2, 0, axis_pair_dim}})->copy_from(runtime, x->narrow({{2, axis_pair_dim, axis_pair_dim}}));
        col->narrow({{2, axis_pair_dim, axis_pair_dim}})->copy_from(runtime, x->narrow({{2, axis_head_dim + axis_pair_dim, axis_pair_dim}}));

        rotary_emb_->forward(runtime, row, row_position_ids, true);
        rotary_emb_->forward(runtime, col, col_position_ids, true);

        x->narrow({{2, 0, axis_pair_dim}})->copy_from(runtime, row->narrow({{2, 0, axis_pair_dim}}));
        x->narrow({{2, axis_head_dim, axis_pair_dim}})->copy_from(runtime, row->narrow({{2, axis_pair_dim, axis_pair_dim}}));
        x->narrow({{2, axis_pair_dim, axis_pair_dim}})->copy_from(runtime, col->narrow({{2, 0, axis_pair_dim}}));
        x->narrow({{2, axis_head_dim + axis_pair_dim, axis_pair_dim}})->copy_from(runtime, col->narrow({{2, axis_pair_dim, axis_pair_dim}}));
    };
    apply_2d_rope(q);
    apply_2d_rope(k);

    const infinilm::runtime::Tensor mha_query = q->unsqueeze(0);
    const infinilm::runtime::Tensor mha_key = k->unsqueeze(0);
    const infinilm::runtime::Tensor mha_value = v;
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
    auto out = mha_output->view({seq_len, hidden_size_});
    return proj_->forward(runtime, out);
}

Ernie45VisionMLP::Ernie45VisionMLP(infinilm::runtime::Runtime &runtime,
                                   const nlohmann::json &config,
                                   const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Ernie45 vision construction requires eager Runtime");
    }
    try {
        const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
        const double hidden_dim_value = static_cast<double>(embed_dim) * config.value("mlp_ratio", 4.0);
        if (!std::isfinite(hidden_dim_value) || hidden_dim_value < 1.0
            || hidden_dim_value >= static_cast<double>(std::numeric_limits<size_t>::max())) {
            throw std::invalid_argument("Ernie45VisionMLP requires representable positive hidden_dim");
        }
        const size_t hidden_dim = static_cast<size_t>(hidden_dim_value);
        fc1_ = std::make_shared<std::remove_reference_t<decltype(*fc1_)>>(runtime, embed_dim, hidden_dim, true, dtype);
        this->bind_child("fc1", fc1_);
        fc2_ = std::make_shared<std::remove_reference_t<decltype(*fc2_)>>(runtime, hidden_dim, embed_dim, true, dtype);
        this->bind_child("fc2", fc2_);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Ernie45VisionMLP::forward(infinilm::runtime::Runtime &runtime,
                                                    const infinilm::runtime::Tensor &hidden_states) const {
    auto x = fc1_->forward(runtime, const_cast<infinilm::runtime::Tensor &>(hidden_states));
    runtime.assertOwnerThread();
    if (!x || runtime.device() != x->device()) {
        throw std::invalid_argument("QuickGelu requires nonnull input on the Runtime device");
    }
    const auto quick_gelu_config = infini::ops::QuickGelu::DefaultConfig(runtime.device().type());
    auto quick_gelu_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::QuickGelu>(
        runtime, infinilm::execution::RecordingPolicy::HostOnly,
        quick_gelu_config, x, quick_gelu_output);
    x = std::move(quick_gelu_output);
    return fc2_->forward(runtime, x);
}

Ernie45VisionBlock::Ernie45VisionBlock(infinilm::runtime::Runtime &runtime,
                                       const nlohmann::json &config,
                                       const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Ernie45 vision construction requires eager Runtime");
    }
    try {
        const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
        norm1_ = std::make_shared<std::remove_reference_t<decltype(*norm1_)>>(runtime, embed_dim, 1e-6, dtype);
        this->bind_child("norm1", norm1_);
        norm2_ = std::make_shared<std::remove_reference_t<decltype(*norm2_)>>(runtime, embed_dim, 1e-6, dtype);
        this->bind_child("norm2", norm2_);
        attn_ = std::make_shared<std::remove_reference_t<decltype(*attn_)>>(runtime, config, dtype);
        this->bind_child("attn", attn_);
        mlp_ = std::make_shared<std::remove_reference_t<decltype(*mlp_)>>(runtime, config, dtype);
        this->bind_child("mlp", mlp_);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Ernie45VisionBlock::forward(infinilm::runtime::Runtime &runtime,
                                                      const infinilm::runtime::Tensor &hidden_states,
                                                      const infinilm::runtime::Tensor &row_position_ids,
                                                      const infinilm::runtime::Tensor &col_position_ids) const {
    auto attn_out = attn_->forward(runtime, norm1_->forward(runtime, hidden_states), row_position_ids, col_position_ids);
    auto hidden_after_attn = infinilm::runtime::Tensor::empty(runtime, hidden_states->shape(), hidden_states->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), hidden_states, attn_out, hidden_after_attn);
    INFINILM_TRACE_OP(Add, hidden_after_attn);
    auto mlp_out = mlp_->forward(runtime, norm2_->forward(runtime, hidden_after_attn));
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, hidden_after_attn->shape(), hidden_after_attn->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), hidden_after_attn, mlp_out, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

Ernie45VisionModel::Ernie45VisionModel(infinilm::runtime::Runtime &runtime,
                                       const nlohmann::json &config,
                                       const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Ernie45 vision construction requires eager Runtime");
    }
    try {
        const size_t depth = config.value("depth", 32);
        const size_t embed_dim = config.value("embed_dim", config.value("hidden_size", 1280));
        spatial_merge_size_ = config.value("spatial_merge_size", 2);
        patch_embed_ = std::make_shared<std::remove_reference_t<decltype(*patch_embed_)>>(runtime, config, dtype);
        this->bind_child("patch_embed", patch_embed_);
        blocks_.reserve(depth);
        for (size_t i = 0; i < depth; ++i) {
            blocks_.push_back(std::make_shared<Ernie45VisionBlock>(runtime, config, dtype));
            this->bind_child("blocks." + std::to_string(i), blocks_.back());
        }
        ln_ = std::make_shared<std::remove_reference_t<decltype(*ln_)>>(runtime, embed_dim, 1e-6, dtype);
        this->bind_child("ln", ln_);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Ernie45VisionModel::build_rotary_position_ids(infinilm::runtime::Runtime &runtime,
                                                                        const infinilm::runtime::Tensor &grid_thw) const {
    auto grid = tensor_to_i64_vector(runtime, grid_thw);
    if (grid.size() % 3 != 0) {
        throw std::runtime_error("Ernie45VisionModel: grid_thw must have shape [3] or [num_images, 3]");
    }

    size_t total_tokens = 0;
    for (size_t i = 0; i < grid.size(); i += 3) {
        total_tokens += static_cast<size_t>(grid[i]) * static_cast<size_t>(grid[i + 1]) * static_cast<size_t>(grid[i + 2]);
    }

    auto position_storage = std::make_shared<std::vector<int64_t>>(2 * total_tokens);
    auto position_ids_cpu = infinilm::runtime::Tensor::from_blob(
        position_storage->data(), {2, total_tokens}, infinilm::runtime::DataType::kInt64,
        infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu}, position_storage);
    auto *position_ids = position_storage->data();
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
    if (grid_thw->device().type() == infinilm::runtime::Device::Type::kCpu) {
        return position_ids_cpu;
    }
    auto position_ids_device = infinilm::runtime::Tensor::empty(runtime, {2, total_tokens}, infinilm::runtime::DataType::kInt64);
    position_ids_device->copy_from(runtime, position_ids_cpu);
    return position_ids_device;
}

infinilm::runtime::Tensor Ernie45VisionModel::forward(infinilm::runtime::Runtime &runtime,
                                                      const infinilm::runtime::Tensor &pixel_values,
                                                      const infinilm::runtime::Tensor &grid_thw) const {
    auto hidden_states = patch_embed_->forward(runtime, pixel_values);
    auto position_ids = build_rotary_position_ids(runtime, grid_thw);
    auto row_position_ids = position_ids->narrow({{0, 0, 1}})->view({position_ids->size(1)});
    auto col_position_ids = position_ids->narrow({{0, 1, 1}})->view({position_ids->size(1)});

    for (auto &block : blocks_) {
        hidden_states = block->forward(runtime, hidden_states, row_position_ids, col_position_ids);
    }
    return ln_->forward(runtime, hidden_states);
}

Ernie45ResamplerModel::Ernie45ResamplerModel(infinilm::runtime::Runtime &runtime,
                                             const nlohmann::json &config,
                                             const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("Ernie45 vision construction requires eager Runtime");
    }
    try {
        const size_t pixel_hidden_size = config.value("pixel_hidden_size", 1280);
        const size_t hidden_size = config.value("hidden_size", 2560);
        spatial_conv_size_ = config.value("spatial_conv_size", 2);
        temporal_conv_size_ = config.value("temporal_conv_size", 2);
        const size_t spatial_dim = pixel_hidden_size * spatial_conv_size_ * spatial_conv_size_;
        const size_t temporal_dim = spatial_dim * temporal_conv_size_;
        use_temporal_conv_ = config.value("use_temporal_conv", true);

        spatial_linear_0_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, spatial_dim, spatial_dim, true, dtype);
        this->bind_child("spatial_linear.0", spatial_linear_0_);
        spatial_linear_2_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, spatial_dim, spatial_dim, true, dtype);
        this->bind_child("spatial_linear.2", spatial_linear_2_);
        spatial_linear_3_ = std::make_shared<Ernie45VisionLayerNorm>(runtime, spatial_dim, 1e-6, dtype);
        this->bind_child("spatial_linear.3", spatial_linear_3_);
        if (use_temporal_conv_) {
            temporal_linear_0_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, temporal_dim, spatial_dim, true, dtype);
            this->bind_child("temporal_linear.0", temporal_linear_0_);
            temporal_linear_2_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, spatial_dim, spatial_dim, true, dtype);
            this->bind_child("temporal_linear.2", temporal_linear_2_);
            temporal_linear_3_ = std::make_shared<Ernie45VisionLayerNorm>(runtime, spatial_dim, 1e-6, dtype);
            this->bind_child("temporal_linear.3", temporal_linear_3_);
        }
        mlp_ = std::make_shared<std::remove_reference_t<decltype(*mlp_)>>(runtime, spatial_dim, hidden_size, true, dtype);
        this->bind_child("mlp", mlp_);
        after_norm_ = std::make_shared<std::remove_reference_t<decltype(*after_norm_)>>(runtime, hidden_size, config.value("rms_norm_eps", 1e-5), dtype);
        this->bind_child("after_norm", after_norm_);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Ernie45ResamplerModel::forward(infinilm::runtime::Runtime &runtime,
                                                         const infinilm::runtime::Tensor &image_features,
                                                         const infinilm::runtime::Tensor &grid_thw) const {
    auto grid = tensor_to_i64_vector(runtime, grid_thw);
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
    x = spatial_linear_0_->forward(runtime, x);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
        infinilm::execution::submit<infini::ops::Gelu>(
            runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), x, std::string{"none"}, ops_output);
        x = std::move(ops_output);
    }
    x = spatial_linear_2_->forward(runtime, x);
    x = spatial_linear_3_->forward(runtime, x);

    if (use_temporal_conv_) {
        if (temporal_conv_size_ != 2) {
            throw std::runtime_error("Ernie45ResamplerModel: temporary temporal packing fallback only supports temporal_conv_size=2");
        }
        const size_t spatial_tokens = (grid_h * grid_w) / spatial_group;
        if (x->size(0) != grid_t * spatial_tokens) {
            throw std::runtime_error("Ernie45ResamplerModel: spatial token length mismatch");
        }

        std::vector<infinilm::runtime::Tensor> first_frames;
        std::vector<infinilm::runtime::Tensor> second_frames;
        for (size_t t = 0; t < grid_t; t += 2) {
            const size_t first = t;
            const size_t second = (t + 1 < grid_t) ? (t + 1) : t;
            first_frames.push_back(x->narrow({{0, first * spatial_tokens, spatial_tokens}}));
            second_frames.push_back(x->narrow({{0, second * spatial_tokens, spatial_tokens}}));
        }
        auto x_first = cat_or_single(runtime, std::move(first_frames), 0);
        auto x_second = cat_or_single(runtime, std::move(second_frames), 0);
        const std::vector<infinilm::runtime::Tensor> merged_parts{x_first, x_second};
        auto merged_shape = merged_parts[0]->shape();
        merged_shape[1] = merged_parts[0]->size(1) + merged_parts[1]->size(1);
        auto merged_tokens = infinilm::runtime::Tensor::empty(runtime, merged_shape, merged_parts[0]->dtype());
        infinilm::execution::tensor_checks::cat::validate(runtime, merged_tokens, merged_parts, {nullptr, 1, 0});
        if (merged_tokens->numel()) {
            infinilm::execution::submit<infini::ops::Cat>(runtime, infini::ops::Cat::DefaultConfig(runtime.device().type()), merged_parts, static_cast<int64_t>(1), merged_tokens);
        }
        x = merged_tokens;
        x = temporal_linear_0_->forward(runtime, x);
        {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
            infinilm::execution::submit<infini::ops::Gelu>(
                runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), x, std::string{"none"}, ops_output);
            x = std::move(ops_output);
        }
        x = temporal_linear_2_->forward(runtime, x);
        x = temporal_linear_3_->forward(runtime, x);
    }

    x = mlp_->forward(runtime, x);
    return after_norm_->forward(runtime, x);
}

} // namespace infinilm::models::ernie4_5_vl
