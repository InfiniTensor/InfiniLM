#include <base/flash_attn_with_kvcache.h>
#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/linear/dense.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include "qwen3_5_vision.hpp"

#include "../../utils.hpp"
#include "runtime/runtime.hpp"
#include <cstring>
#include <memory>
#include <exception>

#include <base/upsample_bilinear2d.h>

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
void check_vision_runtime(infinilm::runtime::Runtime &runtime) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Qwen35 vision construction requires eager Runtime");
}


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

std::vector<int64_t> tensor_to_i64_vector(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &tensor) {
    if (!tensor || !tensor->is_contiguous()) {
        throw std::invalid_argument("Qwen35VisionModel: grid_thw must be contiguous");
    }
    if (tensor->dtype() != infinilm::runtime::DataType::kInt32 &&
        tensor->dtype() != infinilm::runtime::DataType::kInt64) {
        throw std::runtime_error("Qwen35VisionModel: grid_thw must be int32 or int64");
    }
    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    auto read = [&](void *bytes) {
        if (tensor->device() == cpu) {
            std::memcpy(bytes, tensor->data(), tensor->nbytes());
        } else if (tensor->device() == runtime.device()) {
            runtime.memcpyD2H(bytes, tensor->data(), tensor->nbytes());
            runtime.syncStream();
        } else {
            throw std::invalid_argument("Qwen35VisionModel: grid belongs to another device");
        }
    };
    std::vector<int64_t> values(tensor->numel());
    if (tensor->dtype() == infinilm::runtime::DataType::kInt64) {
        read(values.data());
    } else {
        std::vector<int32_t> narrow(tensor->numel());
        read(narrow.data());
        std::copy(narrow.begin(), narrow.end(), values.begin());
    }
    for (auto value : values) {
        if (value <= 0) {
            throw std::invalid_argument("Qwen35VisionModel: grid dimensions must be positive");
        }
        }
        return values;
}

} // namespace

Qwen35VisionPatchProj::Qwen35VisionPatchProj(infinilm::runtime::Runtime &runtime, size_t in_channels,
                                             size_t hidden_size,
                                             size_t temporal_patch_size,
                                             size_t patch_size,
                                             const infinilm::runtime::DataType &dtype)
    : in_channels_(in_channels),
      hidden_size_(hidden_size),
      temporal_patch_size_(temporal_patch_size),
      patch_size_(patch_size) {
    check_vision_runtime(runtime);
    if (!in_channels_ || !hidden_size_ || !temporal_patch_size_ || !patch_size_)
        throw std::invalid_argument("Qwen35 vision patch dimensions must be positive");
    try {
        weight_ = infinilm::weights::Weight(runtime, {hidden_size_, in_channels_, temporal_patch_size_, patch_size_, patch_size_}, dtype);
        this->bind_weight("weight", weight_);
        bias_ = infinilm::weights::Weight(runtime, {hidden_size_}, dtype);
        this->bind_weight("bias", bias_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Qwen35VisionPatchProj::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    runtime.assertOwnerThread();
    const size_t patch_dim = in_channels_ * temporal_patch_size_ * patch_size_ * patch_size_;
    if (hidden_states->shape().size() != 2 || hidden_states->size(1) != patch_dim) {
        throw std::runtime_error("Qwen35VisionPatchProj: expected pixel_values shape [num_patches, patch_dim]");
    }
    auto weight_2d = weight_->view({hidden_size_, patch_dim});
    return infinilm::layers::linear::dense_projection(runtime, hidden_states, weight_2d, std::make_optional<infinilm::runtime::Tensor>(bias_.tensor()));
}

Qwen35VisionPatchEmbed::Qwen35VisionPatchEmbed(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                               const infinilm::runtime::DataType &dtype) {
    check_vision_runtime(runtime);
    try {
    const size_t in_channels = config.value("in_channels", 3);
    const size_t hidden_size = config.value("hidden_size", 1152);
    const size_t temporal_patch_size = get_size_or_first(config, "temporal_patch_size", 2);
    const size_t patch_size = get_size_or_first(config, "patch_size", 16);
        proj_ = this->bind_child("proj", std::make_shared<std::remove_reference_t<decltype(*proj_ )>>(runtime, in_channels, hidden_size, temporal_patch_size, patch_size, dtype));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Qwen35VisionPatchEmbed::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    runtime.assertOwnerThread();
    return proj_->forward(runtime, hidden_states);
}

Qwen35VisionAttention::Qwen35VisionAttention(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                             const infinilm::runtime::DataType &dtype) {
    check_vision_runtime(runtime);
    try {
        hidden_size_ = config.value("hidden_size", 1152);
        num_heads_ = config.value("num_heads", 16);
        if (hidden_size_ == 0 || num_heads_ == 0) throw std::invalid_argument("Qwen35 vision requires positive hidden_size and num_heads");
        head_dim_ = hidden_size_ / num_heads_;
        scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    if (hidden_size_ % num_heads_ != 0) {
        throw std::runtime_error("Qwen35VisionAttention: hidden_size must be divisible by num_heads");
    }
    if (head_dim_ % 4 != 0) {
        throw std::runtime_error("Qwen35VisionAttention: head_dim must be divisible by 4 for 2D RoPE");
    }
    const size_t axis_head_dim = head_dim_ / 2;
        rotary_emb_ = this->bind_child("rotary_emb", std::make_shared<std::remove_reference_t<decltype(*rotary_emb_ )>>(runtime, axis_head_dim, axis_head_dim, 8192, 10000.0, infinilm::layers::RoPE::Algo::GPT_NEOX, dtype, runtime.device()));
        qkv_ = this->bind_child("qkv", std::make_shared<std::remove_reference_t<decltype(*qkv_ )>>(runtime, hidden_size_, hidden_size_ * 3, true, dtype));
        proj_ = this->bind_child("proj", std::make_shared<std::remove_reference_t<decltype(*proj_ )>>(runtime, hidden_size_, hidden_size_, true, dtype));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Qwen35VisionAttention::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                                                  const infinilm::runtime::Tensor &row_position_ids,
                                                  const infinilm::runtime::Tensor &col_position_ids) const {
    runtime.assertOwnerThread();
    const size_t seq_len = hidden_states->size(0);
    const size_t axis_head_dim = head_dim_ / 2;
    auto hidden_mut = hidden_states;
    auto qkv = qkv_->forward(runtime, hidden_mut)->view({seq_len, 3, num_heads_, head_dim_});
    auto q = qkv->narrow({{1, 0, 1}})->squeeze(1)->contiguous(runtime);
    auto k = qkv->narrow({{1, 1, 1}})->squeeze(1)->contiguous(runtime);
    auto v = qkv->narrow({{1, 2, 1}})->squeeze(1)->contiguous(runtime)->unsqueeze(0);

    const size_t axis_pair_dim = axis_head_dim / 2;
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

Qwen35VisionMLP::Qwen35VisionMLP(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                 const infinilm::runtime::DataType &dtype) {
    check_vision_runtime(runtime);
    try {
        activation_ = config.value("hidden_act", "gelu_pytorch_tanh");
    const size_t hidden_size = config.value("hidden_size", 1152);
    const size_t intermediate_size = config.value("intermediate_size", 4304);
        linear_fc1_ = this->bind_child("linear_fc1", std::make_shared<std::remove_reference_t<decltype(*linear_fc1_ )>>(runtime, hidden_size, intermediate_size, true, dtype));
        linear_fc2_ = this->bind_child("linear_fc2", std::make_shared<std::remove_reference_t<decltype(*linear_fc2_ )>>(runtime, intermediate_size, hidden_size, true, dtype));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Qwen35VisionMLP::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    runtime.assertOwnerThread();
    auto hidden_mut = hidden_states;
    auto x = linear_fc1_->forward(runtime, hidden_mut);
    if (activation_ == "gelu" || activation_ == "gelu_pytorch_tanh") {
        x = activation_ == "gelu" ? [&]() {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
            infinilm::execution::submit<infini::ops::Gelu>(
                runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), x, std::string{"none"}, ops_output);
            return ops_output;
        }() : [&]() {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
            infinilm::execution::submit<infini::ops::Gelu>(
                runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), x, std::string{"tanh"}, ops_output);
            return ops_output;
        }();
    } else {
        throw std::runtime_error("Qwen35VisionMLP: unsupported activation " + activation_);
    }
    return linear_fc2_->forward(runtime, x);
}

Qwen35VisionBlock::Qwen35VisionBlock(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                     const infinilm::runtime::DataType &dtype) {
    check_vision_runtime(runtime);
    try {
    const size_t hidden_size = config.value("hidden_size", 1152);
    const double norm_eps = config.value("layer_norm_eps", config.value("rms_norm_eps", 1e-6));
        norm1_ = this->bind_child("norm1", std::make_shared<std::remove_reference_t<decltype(*norm1_ )>>(runtime, hidden_size, norm_eps, dtype));
        norm2_ = this->bind_child("norm2", std::make_shared<std::remove_reference_t<decltype(*norm2_ )>>(runtime, hidden_size, norm_eps, dtype));
        attn_ = this->bind_child("attn", std::make_shared<std::remove_reference_t<decltype(*attn_ )>>(runtime, config, dtype));
        mlp_ = this->bind_child("mlp", std::make_shared<std::remove_reference_t<decltype(*mlp_ )>>(runtime, config, dtype));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Qwen35VisionBlock::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                                              const infinilm::runtime::Tensor &row_position_ids,
                                              const infinilm::runtime::Tensor &col_position_ids) const {
    runtime.assertOwnerThread();
    auto x = norm1_->forward(runtime, hidden_states);
    x = attn_->forward(runtime, x, row_position_ids, col_position_ids);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), x, hidden_states, ops_output);
        INFINILM_TRACE_OP(Add, ops_output);
        x = std::move(ops_output);
    }
    auto residual = x;
    x = norm2_->forward(runtime, x);
    x = mlp_->forward(runtime, x);
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), x, residual, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

Qwen35VisionPatchMerger::Qwen35VisionPatchMerger(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                                 const infinilm::runtime::DataType &dtype) {
    check_vision_runtime(runtime);
    try {
        hidden_size_ = config.value("hidden_size", 1152);
        merged_size_ = hidden_size_ * config.value("spatial_merge_size", 2) * config.value("spatial_merge_size", 2);
    const size_t out_hidden_size = config.value("out_hidden_size", hidden_size_);
    const double norm_eps = config.value("layer_norm_eps", config.value("rms_norm_eps", 1e-6));
        norm_ = this->bind_child("norm", std::make_shared<std::remove_reference_t<decltype(*norm_ )>>(runtime, hidden_size_, norm_eps, dtype));
        linear_fc1_ = this->bind_child("linear_fc1", std::make_shared<std::remove_reference_t<decltype(*linear_fc1_ )>>(runtime, merged_size_, merged_size_, true, dtype));
        linear_fc2_ = this->bind_child("linear_fc2", std::make_shared<std::remove_reference_t<decltype(*linear_fc2_ )>>(runtime, merged_size_, out_hidden_size, true, dtype));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Qwen35VisionPatchMerger::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    runtime.assertOwnerThread();
    auto x = norm_->forward(runtime, hidden_states)->view({hidden_states->size(0) / (merged_size_ / hidden_size_), merged_size_});
    x = linear_fc1_->forward(runtime, x);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
        infinilm::execution::submit<infini::ops::Gelu>(
            runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), x, std::string{"none"}, ops_output);
        x = std::move(ops_output);
    }
    return linear_fc2_->forward(runtime, x);
}

Qwen35VisionModel::Qwen35VisionModel(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                     const infinilm::runtime::DataType &dtype) {
    check_vision_runtime(runtime);
    try {
        hidden_size_ = config.value("hidden_size", 1152);
        num_heads_ = config.value("num_heads", 16);
        if (hidden_size_ == 0 || num_heads_ == 0) throw std::invalid_argument("Qwen35 vision requires positive hidden_size and num_heads");
        head_dim_ = hidden_size_ / num_heads_;
        spatial_merge_size_ = config.value("spatial_merge_size", 2);
        num_grid_per_side_ = static_cast<size_t>(std::sqrt(static_cast<double>(config.value("num_position_embeddings", 2304))));
    const size_t num_position_embeddings = config.value("num_position_embeddings", 2304);
    const size_t depth = config.value("depth", config.value("num_hidden_layers", 27));

        patch_embed_ = this->bind_child("patch_embed", std::make_shared<std::remove_reference_t<decltype(*patch_embed_ )>>(runtime, config, dtype));
        pos_embed_ = this->bind_child("pos_embed", std::make_shared<std::remove_reference_t<decltype(*pos_embed_ )>>(runtime, num_position_embeddings, hidden_size_, std::nullopt, dtype));
    blocks_.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
            blocks_.push_back(this->bind_child("blocks." + std::to_string(i), std::make_shared<Qwen35VisionBlock>(runtime, config, dtype)));
    }
        merger_ = this->bind_child("merger", std::make_shared<std::remove_reference_t<decltype(*merger_ )>>(runtime, config, dtype));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Qwen35VisionModel::fast_pos_embed_interpolate(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &image_grid_thw) const {
    runtime.assertOwnerThread();
    auto grid = tensor_to_i64_vector(runtime, image_grid_thw);
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
    if (pos_nchw->device() != runtime.device()) {
        throw std::invalid_argument("UpsampleBilinear requires NCHW input on its Runtime device");
    }
    const std::vector<int64_t> output_size{static_cast<int64_t>(grid_h), static_cast<int64_t>(grid_w)};
    if (output_size[0] <= 0 || output_size[1] <= 0) {
        throw std::invalid_argument("UpsampleBilinear output_size must contain positive H and W");
    }
    auto resized = infinilm::runtime::Tensor::empty(runtime,
        {pos_nchw->size(0), pos_nchw->size(1), static_cast<size_t>(output_size[0]), static_cast<size_t>(output_size[1])},
        pos_nchw->dtype());
    infinilm::execution::submit<infini::ops::UpsampleBilinear2d>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly, infini::ops::Config::Automatic(),
        pos_nchw, output_size, true, std::optional<double>{}, std::optional<double>{}, resized);
    auto one_frame = resized->squeeze(0)
                         ->permute({1, 2, 0})
                         ->view({1, grid_h / spatial_merge_size_, spatial_merge_size_, grid_w / spatial_merge_size_, spatial_merge_size_, hidden_size_})
                         ->permute({0, 1, 3, 2, 4, 5})
                         ->contiguous(runtime)
                         ->view({grid_h * grid_w, hidden_size_});
    if (grid_t == 1) {
        return one_frame;
    }

    auto pos_embeds = infinilm::runtime::Tensor::empty(runtime, {grid_t * grid_h * grid_w, hidden_size_}, one_frame->dtype());
    for (size_t t = 0; t < grid_t; ++t) {
        pos_embeds->narrow({{0, t * grid_h * grid_w, grid_h * grid_w}})->copy_from(runtime, one_frame);
    }
    return pos_embeds;
}

infinilm::runtime::Tensor Qwen35VisionModel::build_rotary_position_ids(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &image_grid_thw) const {
    runtime.assertOwnerThread();
    auto grid = tensor_to_i64_vector(runtime, image_grid_thw);
    if (grid.size() % 3 != 0) {
        throw std::runtime_error("Qwen35VisionModel: image_grid_thw must have shape [3] or [num_images, 3]");
    }

    size_t total_tokens = 0;
    for (size_t i = 0; i < grid.size(); i += 3) {
        total_tokens += static_cast<size_t>(grid[i]) * static_cast<size_t>(grid[i + 1]) * static_cast<size_t>(grid[i + 2]);
    }

    auto position_owner = std::make_shared<std::vector<int64_t>>(2 * total_tokens);
    auto *position_ids = position_owner->data();

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
    auto host = infinilm::runtime::Tensor::from_blob(position_owner->data(), {2, total_tokens},
        infinilm::runtime::DataType::kInt64, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu}, position_owner);
    auto device_ids = infinilm::runtime::Tensor::empty(runtime, {2, total_tokens}, infinilm::runtime::DataType::kInt64);
    device_ids->copy_from(runtime, host);
    runtime.syncStream();
    return device_ids;
}

infinilm::runtime::Tensor Qwen35VisionModel::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values,
                                              const infinilm::runtime::Tensor &image_grid_thw) const {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::invalid_argument("Qwen35VisionModel: dynamic visual metadata is not recording-qualified");
    }
    // Read and validate shape metadata before any visual device submission.
    const auto grid = tensor_to_i64_vector(runtime, image_grid_thw);
    if (grid.size() != 3 || spatial_merge_size_ == 0 ||
        static_cast<size_t>(grid[1]) % spatial_merge_size_ != 0 ||
        static_cast<size_t>(grid[2]) % spatial_merge_size_ != 0 ||
        pixel_values->size(0) != static_cast<size_t>(grid[0]) *
            static_cast<size_t>(grid[1]) * static_cast<size_t>(grid[2])) {
        throw std::invalid_argument("Qwen35VisionModel: invalid grid or patch count");
    }
    auto hidden_states = patch_embed_->forward(runtime, pixel_values);
    auto pos_embeds = fast_pos_embed_interpolate(runtime, image_grid_thw);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, hidden_states->shape(), hidden_states->dtype());
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), hidden_states, pos_embeds, ops_output);
        INFINILM_TRACE_OP(Add, ops_output);
        hidden_states = std::move(ops_output);
    }

    auto position_ids = build_rotary_position_ids(runtime, image_grid_thw);
    auto row_position_ids = position_ids->narrow({{0, 0, 1}})->view({position_ids->size(1)});
    auto col_position_ids = position_ids->narrow({{0, 1, 1}})->view({position_ids->size(1)});

    for (const auto &block : blocks_) {
        hidden_states = block->forward(runtime, hidden_states, row_position_ids, col_position_ids);
    }
    return merger_->forward(runtime, hidden_states);
}

} // namespace infinilm::models::qwen3_5
