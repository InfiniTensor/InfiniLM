#include "gemma4_vision.hpp"

#include "../../execution/submit.hpp"
#include "gemma4_ops.hpp"
#include <base/add.h>
#include <base/embedding.h>
#include <base/gemm.h>
#include <base/mul.h>
#include <base/relu.h>
#include <base/softmax.h>
#include <cmath>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::models::gemma4 {
namespace {

// Exact GPU clamp(x, lo, hi) using native slot-0 Add/Mul/Relu (works on CUDA + MetaX C550):
//   y = lo + ReLU(x - lo)   == max(x, lo)
//   z = hi - ReLU(hi - y)   == min(y, hi)
infinilm::runtime::Tensor clamp_tensor_gpu(infinilm::runtime::Runtime &runtime,
                                           const infinilm::runtime::Tensor &x,
                                           const infinilm::runtime::Tensor &lo,
                                           const infinilm::runtime::Tensor &hi) {
    const auto dev = runtime.device().type();
    const auto dtype = x->dtype();
    auto neg_one = ops::make_filled_tensor(runtime, {1}, dtype, -1.0);

    // x - lo = x + (-lo)
    auto neg_lo = ops::mul(runtime, lo, neg_one);
    auto diff_lo = ops::add(runtime, x, neg_lo);
    auto relu_lo = infinilm::runtime::Tensor::empty(runtime, diff_lo->shape(), dtype);
    infinilm::execution::submit<infini::ops::Relu>(
        runtime, infini::ops::Relu::DefaultConfig(dev), diff_lo, relu_lo);
    auto clamped_lo = ops::add(runtime, relu_lo, lo);

    // hi - clamped_lo = hi + (-clamped_lo)
    auto neg_clamped_lo = ops::mul(runtime, clamped_lo, neg_one);
    auto diff_hi = ops::add(runtime, neg_clamped_lo, hi);
    auto relu_hi = infinilm::runtime::Tensor::empty(runtime, diff_hi->shape(), dtype);
    infinilm::execution::submit<infini::ops::Relu>(
        runtime, infini::ops::Relu::DefaultConfig(dev), diff_hi, relu_hi);
    auto neg_relu_hi = ops::mul(runtime, relu_hi, neg_one);
    return ops::add(runtime, neg_relu_hi, hi);
}

} // namespace

Gemma4ClippableLinear::Gemma4ClippableLinear(infinilm::runtime::Runtime &runtime,
                                             size_t in_features,
                                             size_t out_features,
                                             bool use_clipped_linears,
                                             infinilm::runtime::DataType dtype)
    : use_clipped_linears_(use_clipped_linears) {
    runtime.assertOwnerThread();
    linear_ = this->bind_child(
        "linear",
        std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, in_features, out_features, false /* bias */, dtype));
    if (use_clipped_linears_) {
        input_min_ = infinilm::weights::Weight(runtime, {1}, dtype);
        input_max_ = infinilm::weights::Weight(runtime, {1}, dtype);
        output_min_ = infinilm::weights::Weight(runtime, {1}, dtype);
        output_max_ = infinilm::weights::Weight(runtime, {1}, dtype);
        this->bind_weight("input_min", input_min_);
        this->bind_weight("input_max", input_max_);
        this->bind_weight("output_min", output_min_);
        this->bind_weight("output_max", output_max_);
    }
}

infinilm::runtime::Tensor Gemma4ClippableLinear::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x) const {
    auto cur = x;
    if (use_clipped_linears_) {
        cur = clamp_tensor_gpu(runtime, cur, input_min_.tensor(), input_max_.tensor());
    }
    auto out = linear_->forward(runtime, cur);
    if (use_clipped_linears_) {
        out = clamp_tensor_gpu(runtime, out, output_min_.tensor(), output_max_.tensor());
    }
    return out;
}

Gemma4VisionPatchEmbedder::Gemma4VisionPatchEmbedder(infinilm::runtime::Runtime &runtime,
                                                     const nlohmann::json &vision_cfg,
                                                     infinilm::runtime::DataType dtype)
    : hidden_size_(vision_cfg.value("hidden_size", size_t{768})),
      patch_size_(vision_cfg.value("patch_size", size_t{16})),
      position_embedding_size_(vision_cfg.value("position_embedding_size", size_t{10240})) {
    runtime.assertOwnerThread();
    const size_t patch_dim = 3 * patch_size_ * patch_size_;
    input_proj_ = this->bind_child(
        "input_proj",
        std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, patch_dim, hidden_size_, false /* bias */, dtype));
    position_embedding_table_ = infinilm::weights::Weight(
        runtime, {2, position_embedding_size_, hidden_size_}, dtype);
    this->bind_weight("position_embedding_table", position_embedding_table_);
}

infinilm::runtime::Tensor Gemma4VisionPatchEmbedder::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &pixel_values,
    const infinilm::runtime::Tensor &pos_x,
    const infinilm::runtime::Tensor &pos_y) const {
    // Gemma 4 scales pixel_values in [0, 1] via 2 * (pixel_values - 0.5) = 2 * pixel_values - 1.
    auto scaled = ops::scale_out(runtime, pixel_values, 2.0);
    auto neg_one = ops::make_filled_tensor(runtime, {1}, scaled->dtype(), -1.0);
    ops::add_inplace(runtime, scaled, neg_one);

    auto h = input_proj_->forward(runtime, scaled);

    const size_t num_patches = pixel_values->size(0);
    auto table_x = position_embedding_table_.tensor()->narrow({{0, 0, 1}})->squeeze(0);
    auto table_y = position_embedding_table_.tensor()->narrow({{0, 1, 1}})->squeeze(0);

    auto emb_x = infinilm::runtime::Tensor::empty(runtime, {num_patches, hidden_size_}, h->dtype());
    auto emb_y = infinilm::runtime::Tensor::empty(runtime, {num_patches, hidden_size_}, h->dtype());
    const auto emb_cfg = infini::ops::Embedding::DefaultConfig(runtime.device().type());
    infinilm::execution::submit<infini::ops::Embedding>(runtime, emb_cfg, pos_x, table_x, emb_x);
    infinilm::execution::submit<infini::ops::Embedding>(runtime, emb_cfg, pos_y, table_y, emb_y);

    ops::add_inplace(runtime, h, emb_x);
    ops::add_inplace(runtime, h, emb_y);
    return h;
}

Gemma4VisionAttention::Gemma4VisionAttention(infinilm::runtime::Runtime &runtime,
                                             const nlohmann::json &vision_cfg,
                                             std::shared_ptr<Gemma4RoPE> axial_rope,
                                             infinilm::runtime::DataType dtype)
    : hidden_size_(vision_cfg.value("hidden_size", size_t{768})),
      num_heads_(vision_cfg.value("num_attention_heads", size_t{12})),
      num_kv_heads_(vision_cfg.value("num_key_value_heads", size_t{12})),
      head_dim_(vision_cfg.value("head_dim", hidden_size_ / num_heads_)),
      rms_norm_eps_(vision_cfg.value("rms_norm_eps", 1e-6)),
      axial_rope_(std::move(axial_rope)) {
    runtime.assertOwnerThread();
    const bool clipped = vision_cfg.value("use_clipped_linears", false);
    q_proj_ = this->bind_child(
        "q_proj", std::make_shared<Gemma4ClippableLinear>(runtime, hidden_size_, num_heads_ * head_dim_, clipped, dtype));
    k_proj_ = this->bind_child(
        "k_proj", std::make_shared<Gemma4ClippableLinear>(runtime, hidden_size_, num_kv_heads_ * head_dim_, clipped, dtype));
    v_proj_ = this->bind_child(
        "v_proj", std::make_shared<Gemma4ClippableLinear>(runtime, hidden_size_, num_kv_heads_ * head_dim_, clipped, dtype));
    o_proj_ = this->bind_child(
        "o_proj", std::make_shared<Gemma4ClippableLinear>(runtime, num_heads_ * head_dim_, hidden_size_, clipped, dtype));

    q_norm_ = this->bind_child("q_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, head_dim_, rms_norm_eps_, dtype));
    k_norm_ = this->bind_child("k_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, head_dim_, rms_norm_eps_, dtype));
    v_norm_ones_ = ops::make_filled_param(runtime, {head_dim_}, dtype, 1.0);
}

infinilm::runtime::Tensor Gemma4VisionAttention::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &hidden_states,
    const infinilm::runtime::Tensor &pos_x,
    const infinilm::runtime::Tensor &pos_y) const {
    const size_t seq_len = hidden_states->size(0);
    const size_t half_dim = head_dim_ / 2;

    // 2D axial RoPE: split last dim into [0:half_dim] (x-axis) and [half_dim:head_dim] (y-axis),
    // applying standard GPT_NEOX RoPE to each half independently (exact match to HF apply_multidimensional_rope).
    auto apply_2d_axial_rope = [&](infinilm::runtime::Tensor &t, size_t n_heads) {
        auto part_x = t->narrow({{2, 0, half_dim}})->contiguous(runtime);
        auto part_y = t->narrow({{2, half_dim, half_dim}})->contiguous(runtime);
        part_x = axial_rope_->forward(runtime, part_x, pos_x, true);
        part_y = axial_rope_->forward(runtime, part_y, pos_y, true);
        t->narrow({{2, 0, half_dim}})->copy_from(runtime, part_x);
        t->narrow({{2, half_dim, half_dim}})->copy_from(runtime, part_y);
        (void)n_heads;
    };

    auto q = q_proj_->forward(runtime, hidden_states)->view({seq_len, num_heads_, head_dim_});
    q = q_norm_->forward(runtime, q);
    apply_2d_axial_rope(q, num_heads_);

    auto k = k_proj_->forward(runtime, hidden_states)->view({seq_len, num_kv_heads_, head_dim_});
    k = k_norm_->forward(runtime, k);
    apply_2d_axial_rope(k, num_kv_heads_);

    auto v = v_proj_->forward(runtime, hidden_states)->view({seq_len, num_kv_heads_, head_dim_});
    v = ops::rms_norm(runtime, v, v_norm_ones_, rms_norm_eps_);

    // Bidirectional attention with scaling = 1.0 (works natively on CUDA + MetaX C550)
    const size_t ngroup = num_heads_ / num_kv_heads_;
    auto Q = q->permute({1, 0, 2})->contiguous(runtime)->view({num_kv_heads_, ngroup * seq_len, head_dim_});
    auto K = k->permute({1, 0, 2})->contiguous(runtime); // [n_kv, seq_len, head_dim]
    auto V = v->permute({1, 0, 2})->contiguous(runtime); // [n_kv, seq_len, head_dim]
    auto K_t = K->permute({0, 2, 1});                    // [n_kv, head_dim, seq_len]

    const auto gemm_cfg = infini::ops::Gemm::DefaultConfig(runtime.device().type());
    auto scores = infinilm::runtime::Tensor::empty(runtime, {num_kv_heads_, ngroup * seq_len, seq_len}, Q->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        gemm_cfg, Q, K_t,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, scores);

    auto scores_3d = scores->view({num_heads_, seq_len, seq_len});
    infinilm::execution::submit<infini::ops::Softmax>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Softmax::DefaultConfig(runtime.device().type()),
        scores_3d, int64_t{-1}, std::optional<infini::ops::DataType>{}, scores_3d);

    auto out = infinilm::runtime::Tensor::empty(runtime, {num_kv_heads_, ngroup * seq_len, head_dim_}, Q->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        gemm_cfg, scores, V,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, out);

    auto attn_out = out->view({num_heads_, seq_len, head_dim_})
                        ->permute({1, 0, 2})
                        ->contiguous(runtime)
                        ->view({seq_len, num_heads_ * head_dim_});
    return o_proj_->forward(runtime, attn_out);
}

Gemma4VisionMLP::Gemma4VisionMLP(infinilm::runtime::Runtime &runtime,
                                 const nlohmann::json &vision_cfg,
                                 infinilm::runtime::DataType dtype) {
    runtime.assertOwnerThread();
    const size_t hidden = vision_cfg.value("hidden_size", size_t{768});
    const size_t inter = vision_cfg.value("intermediate_size", size_t{3072});
    const bool clipped = vision_cfg.value("use_clipped_linears", false);
    gate_proj_ = this->bind_child("gate_proj", std::make_shared<Gemma4ClippableLinear>(runtime, hidden, inter, clipped, dtype));
    up_proj_ = this->bind_child("up_proj", std::make_shared<Gemma4ClippableLinear>(runtime, hidden, inter, clipped, dtype));
    down_proj_ = this->bind_child("down_proj", std::make_shared<Gemma4ClippableLinear>(runtime, inter, hidden, clipped, dtype));
}

infinilm::runtime::Tensor Gemma4VisionMLP::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x) const {
    auto g = ops::gelu_tanh(runtime, gate_proj_->forward(runtime, x));
    auto u = up_proj_->forward(runtime, x);
    return down_proj_->forward(runtime, ops::mul(runtime, g, u));
}

Gemma4VisionEncoderLayer::Gemma4VisionEncoderLayer(infinilm::runtime::Runtime &runtime,
                                                   const nlohmann::json &vision_cfg,
                                                   std::shared_ptr<Gemma4RoPE> axial_rope,
                                                   infinilm::runtime::DataType dtype) {
    runtime.assertOwnerThread();
    const size_t hidden = vision_cfg.value("hidden_size", size_t{768});
    const double eps = vision_cfg.value("rms_norm_eps", 1e-6);
    self_attn_ = this->bind_child("self_attn", std::make_shared<Gemma4VisionAttention>(runtime, vision_cfg, std::move(axial_rope), dtype));
    mlp_ = this->bind_child("mlp", std::make_shared<Gemma4VisionMLP>(runtime, vision_cfg, dtype));
    input_layernorm_ = this->bind_child("input_layernorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, dtype));
    post_attention_layernorm_ = this->bind_child("post_attention_layernorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, dtype));
    pre_feedforward_layernorm_ = this->bind_child("pre_feedforward_layernorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, dtype));
    post_feedforward_layernorm_ = this->bind_child("post_feedforward_layernorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, dtype));
}

infinilm::runtime::Tensor Gemma4VisionEncoderLayer::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &hidden_states,
    const infinilm::runtime::Tensor &pos_x,
    const infinilm::runtime::Tensor &pos_y) const {
    auto res = hidden_states;
    auto h = input_layernorm_->forward(runtime, hidden_states);
    h = self_attn_->forward(runtime, h, pos_x, pos_y);
    h = post_attention_layernorm_->forward(runtime, h);
    h = ops::add(runtime, res, h);

    res = h;
    h = pre_feedforward_layernorm_->forward(runtime, h);
    h = mlp_->forward(runtime, h);
    h = post_feedforward_layernorm_->forward(runtime, h);
    return ops::add(runtime, res, h);
}

Gemma4VisionEncoder::Gemma4VisionEncoder(infinilm::runtime::Runtime &runtime,
                                         const nlohmann::json &vision_cfg,
                                         infinilm::runtime::DataType dtype) {
    runtime.assertOwnerThread();
    const size_t num_layers = vision_cfg.value("num_hidden_layers", size_t{16});
    const size_t hidden = vision_cfg.value("hidden_size", size_t{768});
    const size_t num_heads = vision_cfg.value("num_attention_heads", size_t{12});
    const size_t head_dim = vision_cfg.value("head_dim", hidden / num_heads);
    double rope_theta = 100.0;
    if (vision_cfg.contains("rope_parameters") && vision_cfg.at("rope_parameters").is_object()) {
        rope_theta = vision_cfg.at("rope_parameters").value("rope_theta", 100.0);
    }
    const size_t max_pos = std::min<size_t>(vision_cfg.value("position_embedding_size", size_t{10240}), 1024);
    axial_rope_ = std::make_shared<Gemma4RoPE>(
        runtime, head_dim / 2, max_pos, rope_theta, 1.0, false /* is_proportional */, dtype);

    layers_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        layers_.push_back(this->bind_child(
            "layers." + std::to_string(i),
            std::make_shared<Gemma4VisionEncoderLayer>(runtime, vision_cfg, axial_rope_, dtype)));
    }
}

infinilm::runtime::Tensor Gemma4VisionEncoder::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &inputs_embeds,
    const infinilm::runtime::Tensor &pos_x,
    const infinilm::runtime::Tensor &pos_y) const {
    auto h = inputs_embeds;
    for (const auto &layer : layers_) {
        h = layer->forward(runtime, h, pos_x, pos_y);
    }
    return h;
}

Gemma4VisionModel::Gemma4VisionModel(infinilm::runtime::Runtime &runtime,
                                     const nlohmann::json &vision_cfg,
                                     infinilm::runtime::DataType dtype)
    : hidden_size_(vision_cfg.value("hidden_size", size_t{768})),
      pooling_kernel_size_(vision_cfg.value("pooling_kernel_size", size_t{3})),
      standardize_(vision_cfg.value("standardize", false)),
      dtype_(dtype) {
    runtime.assertOwnerThread();
    patch_embedder_ = this->bind_child(
        "patch_embedder", std::make_shared<Gemma4VisionPatchEmbedder>(runtime, vision_cfg, dtype));
    encoder_ = this->bind_child(
        "encoder", std::make_shared<Gemma4VisionEncoder>(runtime, vision_cfg, dtype));
    if (standardize_) {
        std_bias_ = infinilm::weights::Weight(runtime, {hidden_size_}, dtype);
        std_scale_ = infinilm::weights::Weight(runtime, {hidden_size_}, dtype);
        this->bind_weight("std_bias", std_bias_);
        this->bind_weight("std_scale", std_scale_);
    }
}

infinilm::runtime::Tensor Gemma4VisionModel::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &pixel_values,
    size_t grid_h,
    size_t grid_w) const {
    runtime.assertOwnerThread();
    const size_t num_patches = grid_h * grid_w;
    if (pixel_values->size(0) != num_patches) {
        throw std::invalid_argument("Gemma4VisionModel: pixel_values rows != grid_h * grid_w");
    }
    if ((grid_h % pooling_kernel_size_) != 0 || (grid_w % pooling_kernel_size_) != 0) {
        throw std::invalid_argument("Gemma4VisionModel: grid_h and grid_w must be divisible by pooling_kernel_size");
    }

    // Build exact (x, y) patch coordinates: x = col (0..grid_w-1), y = row (0..grid_h-1)
    std::vector<int64_t> host_x(num_patches);
    std::vector<int64_t> host_y(num_patches);
    for (size_t r = 0; r < grid_h; ++r) {
        for (size_t c = 0; c < grid_w; ++c) {
            host_x[r * grid_w + c] = static_cast<int64_t>(c);
            host_y[r * grid_w + c] = static_cast<int64_t>(r);
        }
    }
    const auto cpu = infinilm::runtime::Device(infinilm::runtime::Device::Type::kCpu, 0);
    auto pos_x = infinilm::runtime::Tensor::empty(runtime, {num_patches}, infinilm::runtime::DataType::kInt64);
    auto pos_y = infinilm::runtime::Tensor::empty(runtime, {num_patches}, infinilm::runtime::DataType::kInt64);
    pos_x->copy_from(runtime, infinilm::runtime::Tensor::from_blob(host_x.data(), {num_patches}, infinilm::runtime::DataType::kInt64, cpu));
    pos_y->copy_from(runtime, infinilm::runtime::Tensor::from_blob(host_y.data(), {num_patches}, infinilm::runtime::DataType::kInt64, cpu));

    auto h = patch_embedder_->forward(runtime, pixel_values, pos_x, pos_y);
    h = encoder_->forward(runtime, h, pos_x, pos_y); // [grid_h * grid_w, H]

    // 2D k x k spatial average pooling in FP32 + sqrt(hidden_size) scaling (exact match to Gemma4VisionPooler)
    const size_t k = pooling_kernel_size_;
    const size_t out_h = grid_h / k;
    const size_t out_w = grid_w / k;
    const size_t out_tokens = out_h * out_w;
    const size_t k2 = k * k;

    auto h_f32 = ops::cast(runtime, h, infinilm::runtime::DataType::kFloat32);
    // [out_h, k, out_w, k, H] -> permute [out_h, out_w, k, k, H] -> [out_tokens, k*k, H]
    auto grouped = h_f32->view({out_h, k, out_w, k, hidden_size_})
                       ->permute({0, 2, 1, 3, 4})
                       ->contiguous(runtime)
                       ->view({out_tokens, k2, hidden_size_});
    // Batch GEMM: [out_tokens, 1, k*k] @ [out_tokens, k*k, H] -> [out_tokens, 1, H]
    // Combined pool average (1/k^2) * sqrt(hidden_size) in one FP32 GEMM!
    const float pool_scale = static_cast<float>(std::sqrt(static_cast<double>(hidden_size_)) / static_cast<double>(k2));
    auto pool_w = ops::make_filled_tensor(runtime, {out_tokens, 1, k2}, infinilm::runtime::DataType::kFloat32, 1.0);
    auto pooled = infinilm::runtime::Tensor::empty(runtime, {out_tokens, 1, hidden_size_}, infinilm::runtime::DataType::kFloat32);
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Gemm::DefaultConfig(runtime.device().type()),
        pool_w, grouped,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{pool_scale},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, pooled);
    auto out_f32 = pooled->view({out_tokens, hidden_size_});

    if (standardize_) {
        auto bias_f32 = ops::cast(runtime, std_bias_.tensor(), infinilm::runtime::DataType::kFloat32);
        auto scale_f32 = ops::cast(runtime, std_scale_.tensor(), infinilm::runtime::DataType::kFloat32);
        auto neg_bias = ops::scale_out(runtime, bias_f32, -1.0);
        ops::add_inplace(runtime, out_f32, neg_bias);
        ops::mul_inplace(runtime, out_f32, scale_f32);
    }

    return ops::cast(runtime, out_f32, dtype_);
}

Gemma4MultimodalEmbedder::Gemma4MultimodalEmbedder(infinilm::runtime::Runtime &runtime,
                                                   size_t multimodal_hidden_size,
                                                   size_t text_hidden_size,
                                                   double rms_norm_eps,
                                                   infinilm::runtime::DataType dtype)
    : rms_norm_eps_(rms_norm_eps) {
    runtime.assertOwnerThread();
    pre_norm_ones_ = ops::make_filled_param(runtime, {multimodal_hidden_size}, dtype, 1.0);
    embedding_projection_ = this->bind_child(
        "embedding_projection",
        std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, multimodal_hidden_size, text_hidden_size, false /* bias */, dtype));
}

infinilm::runtime::Tensor Gemma4MultimodalEmbedder::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &inputs_embeds) const {
    auto normed = ops::rms_norm(runtime, inputs_embeds, pre_norm_ones_, rms_norm_eps_);
    return embedding_projection_->forward(runtime, normed);
}

} // namespace infinilm::models::gemma4
