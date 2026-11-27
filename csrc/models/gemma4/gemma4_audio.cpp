#include "gemma4_audio.hpp"

#include "../../execution/submit.hpp"
#include "gemma4_ops.hpp"
#include "runtime/custom_types.h"
#include "runtime/utils.hpp"
#include <algorithm>
#include <base/add.h>
#include <base/convolution.h>
#include <base/copy.h>
#include <base/fill.h>
#include <base/gemm.h>
#include <base/layer_norm.h>
#include <base/mul.h>
#include <base/relu.h>
#include <base/sigmoid.h>
#include <base/silu.h>
#include <base/softmax.h>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::models::gemma4 {
namespace {

infinilm::runtime::Tensor silu_tensor(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &x) {
    auto out = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::Silu>(
        runtime, infini::ops::Silu::DefaultConfig(runtime.device().type()), x, out);
    return out;
}

infinilm::runtime::Tensor relu_tensor(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &x) {
    auto out = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::Relu>(
        runtime, infini::ops::Relu::DefaultConfig(runtime.device().type()), x, out);
    return out;
}

infinilm::runtime::Tensor sigmoid_tensor(infinilm::runtime::Runtime &runtime,
                                         const infinilm::runtime::Tensor &x) {
    auto out = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::Sigmoid>(
        runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), x, out);
    return out;
}

} // namespace

Gemma4AudioSubSampleLayer::Gemma4AudioSubSampleLayer(infinilm::runtime::Runtime &runtime,
                                                     size_t in_channels,
                                                     size_t out_channels,
                                                     double norm_eps,
                                                     infinilm::runtime::DataType dtype)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      norm_eps_(norm_eps) {
    runtime.assertOwnerThread();
    conv_weight_ = infinilm::weights::Weight(runtime, {out_channels_, in_channels_, 3, 3}, dtype);
    this->bind_weight("conv.weight", conv_weight_);
    norm_weight_ = infinilm::weights::Weight(runtime, {out_channels_}, dtype);
    this->bind_weight("norm.weight", norm_weight_);
    norm_zero_bias_ = ops::make_filled_param(runtime, {out_channels_}, dtype, 0.0);
}

infinilm::runtime::Tensor Gemma4AudioSubSampleLayer::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x) const {
    const size_t n = x->size(0);
    const size_t h = x->size(2);
    const size_t w = x->size(3);
    const size_t out_h = (h + 1) / 2;
    const size_t out_w = (w + 1) / 2;

    auto conv_out = infinilm::runtime::Tensor::empty(runtime, {n, out_channels_, out_h, out_w}, x->dtype());
    infinilm::execution::submit<infini::ops::Convolution>(
        runtime, infini::ops::Convolution::DefaultConfig(runtime.device().type()),
        x, conv_weight_.tensor(), std::optional<infinilm::runtime::Tensor>{},
        std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1}, std::vector<int64_t>{1, 1},
        false, std::vector<int64_t>{0, 0}, int64_t{1}, conv_out);

    // LayerNorm over channels-last [n * out_h * out_w, out_channels]
    auto nhwc = conv_out->permute({0, 2, 3, 1})->contiguous(runtime);
    auto flat = nhwc->view({n * out_h * out_w, out_channels_});
    auto normed = infinilm::runtime::Tensor::empty(runtime, flat->shape(), flat->dtype());
    infinilm::execution::submit<infini::ops::LayerNorm>(
        runtime, infini::ops::LayerNorm::DefaultConfig(runtime.device().type()),
        flat, norm_weight_.tensor(), std::optional<infinilm::runtime::Tensor>{norm_zero_bias_},
        static_cast<float>(norm_eps_), normed);
    auto act = relu_tensor(runtime, normed);
    return act->view({n, out_h, out_w, out_channels_})->permute({0, 3, 1, 2})->contiguous(runtime);
}

Gemma4AudioSubSampleConvProjection::Gemma4AudioSubSampleConvProjection(
    infinilm::runtime::Runtime &runtime,
    const nlohmann::json &audio_cfg,
    infinilm::runtime::DataType dtype) {
    runtime.assertOwnerThread();
    const auto channels = audio_cfg.at("subsampling_conv_channels").get<std::vector<size_t>>();
    const double eps = audio_cfg.value("rms_norm_eps", 1e-6);
    const size_t hidden = audio_cfg.value("hidden_size", size_t{1024});
    layer0_ = this->bind_child("layer0", std::make_shared<Gemma4AudioSubSampleLayer>(runtime, 1, channels[0], eps, dtype));
    layer1_ = this->bind_child("layer1", std::make_shared<Gemma4AudioSubSampleLayer>(runtime, channels[0], channels[1], eps, dtype));
    const size_t proj_in = (channels[0] / 4) * channels[1];
    input_proj_linear_ = this->bind_child(
        "input_proj_linear",
        std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, proj_in, hidden, false /* bias */, dtype));
}

infinilm::runtime::Tensor Gemma4AudioSubSampleConvProjection::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &input_features) const {
    const size_t t = input_features->size(0);
    const size_t f = input_features->size(1);
    auto x = input_features->contiguous(runtime)->view({1, 1, t, f});
    x = layer0_->forward(runtime, x);
    x = layer1_->forward(runtime, x);
    const size_t t_sub = x->size(2);
    const size_t f_sub = x->size(3);
    const size_t c_sub = x->size(1);
    auto flat = x->permute({0, 2, 3, 1})->contiguous(runtime)->view({t_sub, f_sub * c_sub});
    return input_proj_linear_->forward(runtime, flat);
}

Gemma4AudioFeedForward::Gemma4AudioFeedForward(infinilm::runtime::Runtime &runtime,
                                               const nlohmann::json &audio_cfg,
                                               infinilm::runtime::DataType dtype)
    : residual_weight_(audio_cfg.value("residual_weight", 0.5)),
      gradient_clipping_(audio_cfg.value("gradient_clipping", 1e10)) {
    runtime.assertOwnerThread();
    const size_t hidden = audio_cfg.value("hidden_size", size_t{1024});
    const bool clipped = audio_cfg.value("use_clipped_linears", false);
    const double eps = audio_cfg.value("rms_norm_eps", 1e-6);
    ffw_layer_1_ = this->bind_child("ffw_layer_1", std::make_shared<Gemma4ClippableLinear>(runtime, hidden, hidden * 4, clipped, dtype));
    ffw_layer_2_ = this->bind_child("ffw_layer_2", std::make_shared<Gemma4ClippableLinear>(runtime, hidden * 4, hidden, clipped, dtype));
    pre_layer_norm_ = this->bind_child("pre_layer_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, dtype));
    post_layer_norm_ = this->bind_child("post_layer_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, dtype));
}

infinilm::runtime::Tensor Gemma4AudioFeedForward::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x) const {
    auto h = pre_layer_norm_->forward(runtime, x);
    h = ffw_layer_1_->forward(runtime, h);
    h = silu_tensor(runtime, h);
    h = ffw_layer_2_->forward(runtime, h);
    h = post_layer_norm_->forward(runtime, h);
    ops::scale_inplace(runtime, h, residual_weight_);
    return ops::add(runtime, x, h);
}

Gemma4AudioLightConv1d::Gemma4AudioLightConv1d(infinilm::runtime::Runtime &runtime,
                                               const nlohmann::json &audio_cfg,
                                               infinilm::runtime::DataType dtype)
    : hidden_size_(audio_cfg.value("hidden_size", size_t{1024})),
      kernel_size_(audio_cfg.value("conv_kernel_size", size_t{5})) {
    runtime.assertOwnerThread();
    const bool clipped = audio_cfg.value("use_clipped_linears", false);
    const double eps = audio_cfg.value("rms_norm_eps", 1e-6);
    linear_start_ = this->bind_child("linear_start", std::make_shared<Gemma4ClippableLinear>(runtime, hidden_size_, hidden_size_ * 2, clipped, dtype));
    linear_end_ = this->bind_child("linear_end", std::make_shared<Gemma4ClippableLinear>(runtime, hidden_size_, hidden_size_, clipped, dtype));
    depthwise_conv1d_weight_ = infinilm::weights::Weight(runtime, {hidden_size_, 1, kernel_size_}, dtype);
    this->bind_weight("depthwise_conv1d.weight", depthwise_conv1d_weight_);
    pre_layer_norm_ = this->bind_child("pre_layer_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size_, eps, dtype));
    conv_norm_ = this->bind_child("conv_norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size_, eps, dtype));
}

infinilm::runtime::Tensor Gemma4AudioLightConv1d::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x) const {
    const size_t seq_len = x->size(0);
    auto h = pre_layer_norm_->forward(runtime, x);
    h = linear_start_->forward(runtime, h); // [seq_len, 2 * H]

    // GLU(h, dim=-1) = h[:, :H] * sigmoid(h[:, H:])
    auto a = h->narrow({{1, 0, hidden_size_}})->contiguous(runtime);
    auto b = h->narrow({{1, hidden_size_, hidden_size_}})->contiguous(runtime);
    h = ops::mul(runtime, a, sigmoid_tensor(runtime, b)); // [seq_len, H]

    // Causal left-pad by (kernel_size - 1) and run depthwise Conv1d [1, H, seq_len + left_pad] -> [1, H, seq_len]
    const size_t left_pad = kernel_size_ - 1;
    auto h_ct = h->permute({1, 0})->contiguous(runtime); // [H, seq_len]
    auto padded = ops::make_filled_tensor(runtime, {1, hidden_size_, seq_len + left_pad}, h->dtype(), 0.0);
    auto dst = padded->squeeze(0)->narrow({{1, left_pad, seq_len}});
    infinilm::execution::submit<infini::ops::Copy>(
        runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()), h_ct, false, dst);

    auto conv_out = infinilm::runtime::Tensor::empty(runtime, {1, hidden_size_, seq_len}, h->dtype());
    infinilm::execution::submit<infini::ops::Convolution>(
        runtime, infini::ops::Convolution::DefaultConfig(runtime.device().type()),
        padded, depthwise_conv1d_weight_.tensor(), std::optional<infinilm::runtime::Tensor>{},
        std::vector<int64_t>{1}, std::vector<int64_t>{0}, std::vector<int64_t>{1},
        false, std::vector<int64_t>{0}, static_cast<int64_t>(hidden_size_), conv_out);

    h = conv_out->squeeze(0)->permute({1, 0})->contiguous(runtime); // [seq_len, H]
    h = conv_norm_->forward(runtime, h);
    h = silu_tensor(runtime, h);
    h = linear_end_->forward(runtime, h);
    return ops::add(runtime, x, h);
}

Gemma4AudioAttention::Gemma4AudioAttention(infinilm::runtime::Runtime &runtime,
                                           const nlohmann::json &audio_cfg,
                                           infinilm::runtime::DataType dtype)
    : hidden_size_(audio_cfg.value("hidden_size", size_t{1024})),
      num_heads_(audio_cfg.value("num_attention_heads", size_t{8})),
      head_dim_(hidden_size_ / num_heads_),
      chunk_size_(audio_cfg.value("attention_chunk_size", size_t{12})),
      max_past_horizon_(audio_cfg.value("attention_context_left", size_t{13}) - 1),
      max_future_horizon_(audio_cfg.value("attention_context_right", size_t{0})),
      context_size_(chunk_size_ + max_past_horizon_ + max_future_horizon_),
      softcap_(audio_cfg.value("attention_logit_cap", 50.0)),
      invalid_logits_value_(audio_cfg.value("attention_invalid_logits_value", -1e9)) {
    runtime.assertOwnerThread();
    q_scale_ = (1.0 / std::sqrt(static_cast<double>(head_dim_))) / std::log(2.0);
    k_scale_ = std::log(1.0 + std::exp(1.0)) / std::log(2.0);

    const bool clipped = audio_cfg.value("use_clipped_linears", false);
    const size_t qkv_dim = num_heads_ * head_dim_;
    q_proj_ = this->bind_child("q_proj", std::make_shared<Gemma4ClippableLinear>(runtime, hidden_size_, qkv_dim, clipped, dtype));
    k_proj_ = this->bind_child("k_proj", std::make_shared<Gemma4ClippableLinear>(runtime, hidden_size_, qkv_dim, clipped, dtype));
    v_proj_ = this->bind_child("v_proj", std::make_shared<Gemma4ClippableLinear>(runtime, hidden_size_, qkv_dim, clipped, dtype));
    post_ = this->bind_child("post", std::make_shared<Gemma4ClippableLinear>(runtime, hidden_size_, hidden_size_, clipped, dtype));
    relative_k_proj_ = this->bind_child(
        "relative_k_proj",
        std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, hidden_size_, qkv_dim, false /* bias */, dtype));
    per_dim_scale_ = infinilm::weights::Weight(runtime, {head_dim_}, dtype);
    this->bind_weight("per_dim_scale", per_dim_scale_);
}

infinilm::runtime::Tensor Gemma4AudioAttention::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x,
    const infinilm::runtime::Tensor &rel_pos_embed) const {
    const size_t seq_len = x->size(0);
    const auto model_dtype = x->dtype();
    const auto f32 = infinilm::runtime::DataType::kFloat32;

    // Project Q, K, V and upcast to FP32 (matching HF Gemma4AudioAttention.forward)
    auto q = ops::cast(runtime, q_proj_->forward(runtime, x), f32)->view({seq_len, num_heads_, head_dim_});
    auto k = ops::cast(runtime, k_proj_->forward(runtime, x), f32)->view({seq_len, num_heads_, head_dim_});
    auto v = ops::cast(runtime, v_proj_->forward(runtime, x), f32)->view({seq_len, num_heads_, head_dim_});

    // Compute q_scale * softplus(per_dim_scale) cleanly on CPU (tiny 128 floats!) and upload to GPU
    auto pds_f32 = ops::cast(runtime, per_dim_scale_.tensor(), f32);
    std::vector<float> pds_host(head_dim_);
    if (runtime.device().type() == infinilm::runtime::Device::Type::kCpu) {
        std::memcpy(pds_host.data(), pds_f32->data(), head_dim_ * sizeof(float));
    } else {
        runtime.memcpyD2H(pds_host.data(), pds_f32->data(), head_dim_ * sizeof(float));
        runtime.syncStream();
    }
    for (size_t d = 0; d < head_dim_; ++d) {
        const double val = static_cast<double>(pds_host[d]);
        const double sp = (val > 20.0) ? val : std::log1p(std::exp(val));
        pds_host[d] = static_cast<float>(q_scale_ * sp);
    }
    const auto cpu = infinilm::runtime::Device(infinilm::runtime::Device::Type::kCpu, 0);
    auto q_dim_scale = infinilm::runtime::Tensor::empty(runtime, {1, 1, head_dim_}, f32);
    q_dim_scale->copy_from(runtime, infinilm::runtime::Tensor::from_blob(pds_host.data(), {1, 1, head_dim_}, f32, cpu));
    ops::mul_inplace(runtime, q, q_dim_scale);
    ops::scale_inplace(runtime, k, k_scale_);

    // Relative keys: rel_pos_embed [pos_len, hidden_size] -> [num_heads, head_dim, pos_len]
    auto rel_mut = rel_pos_embed;
    auto rel_k = ops::cast(runtime, relative_k_proj_->forward(runtime, rel_mut), f32);
    const size_t pos_len = rel_pos_embed->size(0);
    auto rel_k_h = rel_k->view({pos_len, num_heads_, head_dim_})->permute({1, 2, 0})->contiguous(runtime); // [H, D, pos_len]

    // Direct local-window attention over all seq_len tokens (exact match to chunked attention with left/right horizon):
    // For each query position i, valid keys j satisfy:
    //   0 <= j < seq_len,  i - max_past_horizon <= j <= i + max_future_horizon.
    // Using full [H, seq_len, seq_len] GEMM with exact relative position shift + softcap + local mask is fast
    // (audio sequences are short, ~50-400 frames after 4x subsampling!) and 100% native GPU!
    auto Q_h = q->permute({1, 0, 2})->contiguous(runtime); // [H, T, D]
    auto K_h = k->permute({1, 2, 0})->contiguous(runtime); // [H, D, T]
    auto V_h = v->permute({1, 0, 2})->contiguous(runtime); // [H, T, D]

    const auto gemm_cfg = infini::ops::Gemm::DefaultConfig(runtime.device().type());
    auto scores = infinilm::runtime::Tensor::empty(runtime, {num_heads_, seq_len, seq_len}, f32);
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        gemm_cfg, Q_h, K_h,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, scores);

    // Relative bias: Q_h @ rel_k_h -> [H, T, pos_len], where pos index p corresponds to relative distance d = (pos_len - 1) - p = i - j.
    auto bd_raw = infinilm::runtime::Tensor::empty(runtime, {num_heads_, seq_len, pos_len}, f32);
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        gemm_cfg, Q_h, rel_k_h,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, bd_raw);

    // Scatter relative scores into [H, T, T] and apply exact local-window mask on GPU.
    // Note: pos_len = context_size / 2 + 1 (13), so for 0 <= i - j <= max_past_horizon (12),
    // rel_idx = (pos_len - 1) - (i - j) is always in [0, pos_len - 1].
    // Construct a [T, T] mask tensor on host (tiny: e.g. 200x200 = 160 KB) and apply softcap + mask + softmax on GPU.
    std::vector<float> scores_host(num_heads_ * seq_len * seq_len);
    std::vector<float> bd_host(num_heads_ * seq_len * pos_len);
    if (runtime.device().type() == infinilm::runtime::Device::Type::kCpu) {
        std::memcpy(scores_host.data(), scores->data(), scores_host.size() * sizeof(float));
        std::memcpy(bd_host.data(), bd_raw->data(), bd_host.size() * sizeof(float));
    } else {
        runtime.memcpyD2H(scores_host.data(), scores->data(), scores_host.size() * sizeof(float));
        runtime.memcpyD2H(bd_host.data(), bd_raw->data(), bd_host.size() * sizeof(float));
        runtime.syncStream();
    }

    const int64_t left = static_cast<int64_t>(max_past_horizon_);
    const int64_t right = static_cast<int64_t>(max_future_horizon_);
    const int64_t max_rel = static_cast<int64_t>(pos_len) - 1;
    const float inv_cap = static_cast<float>(1.0 / softcap_);
    const float cap_f = static_cast<float>(softcap_);
    const float invalid_f = static_cast<float>(invalid_logits_value_);

    for (size_t h_idx = 0; h_idx < num_heads_; ++h_idx) {
        for (size_t i = 0; i < seq_len; ++i) {
            float *row = scores_host.data() + (h_idx * seq_len + i) * seq_len;
            const float *bd_row = bd_host.data() + (h_idx * seq_len + i) * pos_len;
            for (size_t j = 0; j < seq_len; ++j) {
                const int64_t dist = static_cast<int64_t>(i) - static_cast<int64_t>(j);
                if (dist >= -right && dist <= left) {
                    const int64_t rel_idx = max_rel - dist;
                    float val = row[j];
                    if (rel_idx >= 0 && rel_idx < static_cast<int64_t>(pos_len)) {
                        val += bd_row[rel_idx];
                    }
                    row[j] = cap_f * std::tanh(val * inv_cap);
                } else {
                    row[j] = invalid_f;
                }
            }
        }
    }
    scores->copy_from(runtime, infinilm::runtime::Tensor::from_blob(scores_host.data(), {num_heads_, seq_len, seq_len}, f32, cpu));

    infinilm::execution::submit<infini::ops::Softmax>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Softmax::DefaultConfig(runtime.device().type()),
        scores, int64_t{-1}, std::optional<infini::ops::DataType>{}, scores);

    auto out_f32 = infinilm::runtime::Tensor::empty(runtime, {num_heads_, seq_len, head_dim_}, f32);
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        gemm_cfg, scores, V_h,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
        std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, out_f32);

    auto out_merged = out_f32->permute({1, 0, 2})->contiguous(runtime)->view({seq_len, num_heads_ * head_dim_});
    return post_->forward(runtime, ops::cast(runtime, out_merged, model_dtype));
}

Gemma4AudioLayer::Gemma4AudioLayer(infinilm::runtime::Runtime &runtime,
                                   const nlohmann::json &audio_cfg,
                                   infinilm::runtime::DataType dtype) {
    runtime.assertOwnerThread();
    const size_t hidden = audio_cfg.value("hidden_size", size_t{1024});
    const double eps = audio_cfg.value("rms_norm_eps", 1e-6);
    feed_forward1_ = this->bind_child("feed_forward1", std::make_shared<Gemma4AudioFeedForward>(runtime, audio_cfg, dtype));
    feed_forward2_ = this->bind_child("feed_forward2", std::make_shared<Gemma4AudioFeedForward>(runtime, audio_cfg, dtype));
    self_attn_ = this->bind_child("self_attn", std::make_shared<Gemma4AudioAttention>(runtime, audio_cfg, dtype));
    lconv1d_ = this->bind_child("lconv1d", std::make_shared<Gemma4AudioLightConv1d>(runtime, audio_cfg, dtype));
    norm_pre_attn_ = this->bind_child("norm_pre_attn", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, dtype));
    norm_post_attn_ = this->bind_child("norm_post_attn", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, dtype));
    norm_out_ = this->bind_child("norm_out", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden, eps, dtype));
}

infinilm::runtime::Tensor Gemma4AudioLayer::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &x,
    const infinilm::runtime::Tensor &rel_pos_embed) const {
    auto h = feed_forward1_->forward(runtime, x);
    auto res = h;
    h = norm_pre_attn_->forward(runtime, h);
    h = self_attn_->forward(runtime, h, rel_pos_embed);
    h = norm_post_attn_->forward(runtime, h);
    h = ops::add(runtime, res, h);

    h = lconv1d_->forward(runtime, h);
    h = feed_forward2_->forward(runtime, h);
    return norm_out_->forward(runtime, h);
}

Gemma4AudioModel::Gemma4AudioModel(infinilm::runtime::Runtime &runtime,
                                   const nlohmann::json &audio_cfg,
                                   infinilm::runtime::DataType dtype)
    : hidden_size_(audio_cfg.value("hidden_size", size_t{1024})),
      dtype_(dtype) {
    runtime.assertOwnerThread();
    const size_t chunk = audio_cfg.value("attention_chunk_size", size_t{12});
    const size_t left = audio_cfg.value("attention_context_left", size_t{13});
    const size_t right = audio_cfg.value("attention_context_right", size_t{0});
    context_size_ = chunk + (left - 1) + right;
    const size_t num_layers = audio_cfg.value("num_hidden_layers", size_t{12});
    const size_t output_proj_dims = audio_cfg.value("output_proj_dims", size_t{1536});

    subsample_conv_projection_ = this->bind_child(
        "subsample_conv_projection",
        std::make_shared<Gemma4AudioSubSampleConvProjection>(runtime, audio_cfg, dtype));
    layers_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        layers_.push_back(this->bind_child(
            "layers." + std::to_string(i),
            std::make_shared<Gemma4AudioLayer>(runtime, audio_cfg, dtype)));
    }
    output_proj_ = this->bind_child(
        "output_proj",
        std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, hidden_size_, output_proj_dims, true /* bias */, dtype));

    // Precompute sinusoidal relative positional encodings [context_size/2 + 1, hidden_size]
    const size_t pos_len = context_size_ / 2 + 1;
    const size_t half = hidden_size_ / 2;
    const double log_inc = std::log(10000.0) / static_cast<double>(std::max<size_t>(half - 1, 1));
    std::vector<float> pos_f32(pos_len * hidden_size_);
    for (size_t idx = 0; idx < pos_len; ++idx) {
        const double pos = static_cast<double>(context_size_ / 2 - idx);
        for (size_t j = 0; j < half; ++j) {
            const double angle = pos * std::exp(-static_cast<double>(j) * log_inc);
            pos_f32[idx * hidden_size_ + j] = static_cast<float>(std::sin(angle));
            pos_f32[idx * hidden_size_ + half + j] = static_cast<float>(std::cos(angle));
        }
    }
    const auto cpu = infinilm::runtime::Device(infinilm::runtime::Device::Type::kCpu, 0);
    rel_pos_embed_ = infinilm::runtime::Tensor::empty_parameter(runtime, {pos_len, hidden_size_}, dtype_);
    if (dtype_ == infinilm::runtime::DataType::kFloat32) {
        rel_pos_embed_->copy_from(runtime, infinilm::runtime::Tensor::from_blob(pos_f32.data(), {pos_len, hidden_size_}, dtype_, cpu));
    } else if (dtype_ == infinilm::runtime::DataType::kBFloat16) {
        std::vector<bf16_t> buf(pos_f32.size());
        for (size_t i = 0; i < pos_f32.size(); ++i) {
            buf[i] = ::utils::cast<bf16_t, float>(pos_f32[i]);
        }
        rel_pos_embed_->copy_from(runtime, infinilm::runtime::Tensor::from_blob(buf.data(), {pos_len, hidden_size_}, dtype_, cpu));
    } else {
        std::vector<fp16_t> buf(pos_f32.size());
        for (size_t i = 0; i < pos_f32.size(); ++i) {
            buf[i] = ::utils::cast<fp16_t, float>(pos_f32[i]);
        }
        rel_pos_embed_->copy_from(runtime, infinilm::runtime::Tensor::from_blob(buf.data(), {pos_len, hidden_size_}, dtype_, cpu));
    }
    if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
        runtime.syncStream();
    }
}

infinilm::runtime::Tensor Gemma4AudioModel::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &input_features) const {
    auto h = subsample_conv_projection_->forward(runtime, input_features);
    for (const auto &layer : layers_) {
        h = layer->forward(runtime, h, rel_pos_embed_);
    }
    return output_proj_->forward(runtime, h);
}

} // namespace infinilm::models::gemma4
