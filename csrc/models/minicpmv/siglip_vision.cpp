#include <base/flash_attn_with_kvcache.h>
#include "../../weights/weights.hpp"
#include "../../execution/submit.hpp"
#include <base/gemm.h>
#include <base/convolution.h>
#include "siglip_vision.hpp"


#include "runtime/context.hpp"
#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"
#include <memory>
#include <exception>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <limits>

namespace infinilm::models::minicpmv {

SiglipPatchEmbedding::SiglipPatchEmbedding(infinilm::runtime::Runtime &runtime, size_t in_channels,
                                           size_t out_channels,
                                           size_t patch_size,
                                           const infinilm::runtime::DataType &dtype)
    : patch_size_(patch_size) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("SiglipPatchEmbedding construction requires eager Runtime");
    try {
    weight_ = infinilm::weights::Weight(runtime, {out_channels, in_channels, patch_size_, patch_size_}, dtype); this->bind_weight("weight", weight_);
    bias_ = infinilm::weights::Weight(runtime, {out_channels}, dtype); this->bind_weight("bias", bias_);

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor SiglipPatchEmbedding::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values) const {
    runtime.assertOwnerThread();
    constexpr auto max_extent = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    if (patch_size_ == 0 || patch_size_ > max_extent)
        throw std::invalid_argument("conv2d requires a positive int64 stride");
    const auto weight = weight_.tensor();
    const auto bias = bias_.tensor();
    if (!pixel_values || !weight || pixel_values->ndim() != 4 || weight->ndim() != 4
        || pixel_values->size(1) != weight->size(1))
        throw std::invalid_argument("conv2d requires matching rank-4 input and weight");
    if (pixel_values->device() != runtime.device() || weight->device() != runtime.device()
        || (bias && bias->device() != runtime.device()))
        throw std::invalid_argument("conv2d tensors must belong to the supplied runtime");
    auto shape = pixel_values->shape();
    shape[1] = weight->size(0);
    for (size_t axis = 2; axis < 4; ++axis) {
        const auto kernel = weight->size(axis);
        if (shape[axis] > max_extent || kernel == 0 || kernel > max_extent)
            throw std::invalid_argument("conv2d has invalid spatial dimensions");
        if (shape[axis] < kernel)
            throw std::invalid_argument("conv2d output has a non-positive spatial dimension");
        shape[axis] = (shape[axis] - kernel) / patch_size_ + 1;
    }
    auto output = infinilm::runtime::Tensor::empty(runtime, shape, pixel_values->dtype());
    const std::vector<int64_t> strides(2, static_cast<int64_t>(patch_size_));
    const std::optional<infinilm::runtime::Tensor> ops_bias = bias
        ? std::optional<infinilm::runtime::Tensor>{bias} : std::nullopt;
    infinilm::execution::submit<infini::ops::Convolution>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Convolution::DefaultConfig(runtime.device().type()),
        pixel_values, weight, ops_bias, strides, std::vector<int64_t>{0, 0},
        std::vector<int64_t>{1, 1}, false, std::vector<int64_t>{0, 0}, int64_t{1}, output);
    return output;
}

SiglipVisionEmbeddings::SiglipVisionEmbeddings(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                               const infinilm::runtime::DataType &dtype)
    : hidden_size_(config.value("hidden_size", 768)),
      patch_size_(config.value("patch_size", 16)),
      num_positions_(patch_size_ ? (config.value("image_size", 224) / patch_size_) * (config.value("image_size", 224) / patch_size_) : 0) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("SiglipVisionEmbeddings construction requires eager Runtime");
    if (patch_size_ == 0) throw std::invalid_argument("SiglipVisionEmbeddings requires positive patch_size");
    try {
    patch_embedding_ = this->bind_child("patch_embedding", std::make_shared<std::remove_reference_t<decltype(*patch_embedding_ )>>(runtime, 3, hidden_size_, patch_size_, dtype));
    position_embedding_ = this->bind_child("position_embedding", std::make_shared<std::remove_reference_t<decltype(*position_embedding_ )>>(runtime, num_positions_, hidden_size_, std::nullopt, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor SiglipVisionEmbeddings::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values,
                                                   const infinilm::runtime::Tensor &tgt_sizes) const {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::invalid_argument("SiglipVisionEmbeddings: data-dependent position metadata is not recording-qualified");
    }
    if (!pixel_values || pixel_values->device() != runtime.device() ||
        !tgt_sizes || tgt_sizes->dtype() != infinilm::runtime::DataType::kInt64 ||
        !tgt_sizes->is_contiguous() || tgt_sizes->ndim() != 2 ||
        tgt_sizes->size(0) != pixel_values->size(0) || tgt_sizes->size(1) != 2) {
        throw std::invalid_argument("SiglipVisionEmbeddings: invalid pixels or target-size metadata");
    }
    std::vector<int64_t> tgt_sizes_host(tgt_sizes->numel());
    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu, 0};
    if (tgt_sizes->device() == cpu) {
        std::memcpy(tgt_sizes_host.data(), tgt_sizes->data(), tgt_sizes->nbytes());
    } else if (tgt_sizes->device() == runtime.device()) {
        runtime.memcpyD2H(tgt_sizes_host.data(), tgt_sizes->data(), tgt_sizes->nbytes());
        runtime.syncStream();
    } else {
        throw std::invalid_argument("SiglipVisionEmbeddings: metadata belongs to another device");
    }
    for (auto value : tgt_sizes_host) {
        if (value <= 0) throw std::invalid_argument("SiglipVisionEmbeddings: target sizes must be positive");
    }
    auto patch_embeds = patch_embedding_->forward(runtime, pixel_values);
    auto batch_size = patch_embeds->size(0);
    auto seq_len = patch_embeds->size(2) * patch_embeds->size(3);

    auto embeddings = patch_embeds->view({batch_size, hidden_size_, seq_len})->permute({0, 2, 1});

    // Build position ids on CPU
    auto pos_owner = std::make_shared<std::vector<int64_t>>(batch_size * seq_len, 0);
    auto *pos_ptr = pos_owner->data();
    const size_t num_patches_per_side = static_cast<size_t>(std::sqrt(static_cast<double>(num_positions_)));

    for (size_t b = 0; b < batch_size; ++b) {
        size_t nb_h = num_patches_per_side;
        size_t nb_w = num_patches_per_side;
        if (!tgt_sizes_host.empty()) {
            nb_h = static_cast<size_t>(tgt_sizes_host[b * 2]);
            nb_w = static_cast<size_t>(tgt_sizes_host[b * 2 + 1]);
        }
        size_t patch_len = nb_h * nb_w;
        for (size_t ih = 0; ih < nb_h; ++ih) {
            size_t bh = (ih * num_patches_per_side) / nb_h;
            for (size_t iw = 0; iw < nb_w; ++iw) {
                size_t bw = (iw * num_patches_per_side) / nb_w;
                size_t pos_id = bh * num_patches_per_side + bw;
                size_t idx = ih * nb_w + iw;
                if (idx < seq_len) {
                    pos_ptr[b * seq_len + idx] = static_cast<int64_t>(pos_id);
                }
            }
        }
    }

    auto pos_host = infinilm::runtime::Tensor::from_blob(pos_owner->data(), {batch_size, seq_len},
        infinilm::runtime::DataType::kInt64, cpu, pos_owner);
    auto pos_ids = infinilm::runtime::Tensor::empty(runtime, {batch_size, seq_len}, infinilm::runtime::DataType::kInt64);
    pos_ids->copy_from(runtime, pos_host);
    runtime.syncStream(); // owned host bytes may be released after the explicit upload
    auto pos_embed = position_embedding_->forward(runtime, pos_ids);
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, embeddings->shape(), embeddings->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), embeddings, pos_embed, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

SiglipAttention::SiglipAttention(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                 const infinilm::runtime::DataType &dtype)
    : embed_dim_(config.value("hidden_size", 768)),
      num_heads_(config.value("num_attention_heads", 12)),
      head_dim_(num_heads_ ? embed_dim_ / num_heads_ : 0),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("SiglipAttention construction requires eager Runtime");
    try {
    if (embed_dim_ == 0 || num_heads_ == 0 || embed_dim_ % num_heads_ != 0) {
        throw std::runtime_error("SiglipAttention: embed_dim must be divisible by num_heads");
    }
    qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
        runtime, embed_dim_, head_dim_, num_heads_, num_heads_,
        "q_proj", "k_proj", "v_proj", [this](const std::string &n, infinilm::weights::Weight p) { this->bind_weight(n, std::move(p)); },
        nullptr, true, dtype, engine::distributed::RankInfo(runtime.device()));

    out_proj_ = this->bind_child("out_proj", std::make_shared<std::remove_reference_t<decltype(*out_proj_ )>>(runtime, embed_dim_, embed_dim_, true, dtype));


    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor SiglipAttention::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                                            const std::optional<infinilm::runtime::Tensor> &attention_mask,
                                            backends::AttentionBackend attention_backend) const {
    runtime.assertOwnerThread();
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto qkv = qkv_proj_->forward(runtime, const_cast<infinilm::runtime::Tensor &>(hidden_states))->view({batch_size, seq_len, num_heads_ * 3, head_dim_});
    auto q = qkv->narrow({{2, 0, num_heads_}});
    auto k = qkv->narrow({{2, num_heads_, num_heads_}});
    auto v = qkv->narrow({{2, num_heads_ * 2, num_heads_}});

    if (attention_backend == infinilm::backends::AttentionBackend::FLASH_ATTN) {
        const infinilm::runtime::Tensor mha_query = q;
        const infinilm::runtime::Tensor mha_key = k;
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
        auto out = mha_output->view({batch_size, seq_len, num_heads_ * head_dim_});
        return out_proj_->forward(runtime, out);
    } else {
        auto q_reshaped = q->view({batch_size, seq_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous(runtime);
        auto k_reshaped = k->view({batch_size, seq_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous(runtime);
        auto v_reshaped = v->view({batch_size, seq_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous(runtime);

        auto q_flat = q_reshaped->view({batch_size * num_heads_, seq_len, head_dim_});
        auto k_flat = k_reshaped->view({batch_size * num_heads_, seq_len, head_dim_});
        auto v_flat = v_reshaped->view({batch_size * num_heads_, seq_len, head_dim_});

        auto k_t = k_flat->permute({0, 2, 1});
        auto attn_weights_shape = q_flat->shape();
        attn_weights_shape[q_flat->ndim() - 1] = k_t->size(q_flat->ndim() - 1);
        auto attn_weights = infinilm::runtime::Tensor::empty(runtime, attn_weights_shape, q_flat->dtype());
        infinilm::execution::submit<infini::ops::Gemm>(
            runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
            infini::ops::Gemm::DefaultConfig(runtime.device().type()), q_flat, k_t,
            std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{scale_},
            std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, attn_weights);
        INFINILM_TRACE_OP(Gemm, attn_weights);

        auto attn_view = attn_weights->view({batch_size * num_heads_, seq_len, seq_len});
        infinilm::execution::submit<infini::ops::Softmax>(
            runtime, infini::ops::Softmax::DefaultConfig(runtime.device().type()), attn_view, std::int64_t{-1}, std::optional<infini::ops::DataType>{}, attn_view);

        auto attn_output_shape = attn_weights->shape();
        attn_output_shape[attn_weights->ndim() - 1] = v_flat->size(attn_weights->ndim() - 1);
        auto attn_output = infinilm::runtime::Tensor::empty(runtime, attn_output_shape, attn_weights->dtype());
        infinilm::execution::submit<infini::ops::Gemm>(
            runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
            infini::ops::Gemm::DefaultConfig(runtime.device().type()), attn_weights, v_flat,
            std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
            std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, attn_output);
        INFINILM_TRACE_OP(Gemm, attn_output);
        auto out = attn_output->view({batch_size, num_heads_, seq_len, head_dim_})
                       ->permute({0, 2, 1, 3})
                       ->contiguous(runtime)
                       ->view({batch_size, seq_len, embed_dim_});
        return out_proj_->forward(runtime, out);
    }
}

SiglipMLP::SiglipMLP(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                     const infinilm::runtime::DataType &dtype)
    : activation_(config.value("hidden_act", "gelu")) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("SiglipMLP construction requires eager Runtime");
    try {
    size_t hidden_size = config.value("hidden_size", 768);
    size_t intermediate_size = config.value("intermediate_size", 3072);
    fc1_ = this->bind_child("fc1", std::make_shared<std::remove_reference_t<decltype(*fc1_ )>>(runtime, hidden_size, intermediate_size, true, dtype));
    fc2_ = this->bind_child("fc2", std::make_shared<std::remove_reference_t<decltype(*fc2_ )>>(runtime, intermediate_size, hidden_size, true, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor SiglipMLP::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    runtime.assertOwnerThread();
    auto x = fc1_->forward(runtime, const_cast<infinilm::runtime::Tensor &>(hidden_states));
    if (activation_ == "gelu_tanh") {
        {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
            infinilm::execution::submit<infini::ops::Gelu>(
                runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), x, std::string{"tanh"}, ops_output);
            x = std::move(ops_output);
        }
    } else if (activation_ == "gelu") {
        {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
            infinilm::execution::submit<infini::ops::Gelu>(
                runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), x, std::string{"none"}, ops_output);
            x = std::move(ops_output);
        }
    } else if (activation_ == "relu") {
        {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
            infinilm::execution::submit<infini::ops::Relu>(
                runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
                infini::ops::Relu::DefaultConfig(runtime.device().type()), x, ops_output);
            x = std::move(ops_output);
        }
    } else {
        throw std::runtime_error("SiglipMLP: unsupported activation " + activation_);
    }
    return fc2_->forward(runtime, x);
}

SiglipEncoderLayer::SiglipEncoderLayer(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                       const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("SiglipEncoderLayer construction requires eager Runtime");
    try {
    size_t hidden_size = config.value("hidden_size", 768);
    float layer_norm_eps = config.value("layer_norm_eps", 1e-6f);
    layer_norm1_ = this->bind_child("layer_norm1", std::make_shared<std::remove_reference_t<decltype(*layer_norm1_ )>>(runtime, hidden_size, layer_norm_eps, dtype));
    self_attn_ = this->bind_child("self_attn", std::make_shared<std::remove_reference_t<decltype(*self_attn_ )>>(runtime, config, dtype));
    layer_norm2_ = this->bind_child("layer_norm2", std::make_shared<std::remove_reference_t<decltype(*layer_norm2_ )>>(runtime, hidden_size, layer_norm_eps, dtype));
    mlp_ = this->bind_child("mlp", std::make_shared<std::remove_reference_t<decltype(*mlp_ )>>(runtime, config, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor SiglipEncoderLayer::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                                               const std::optional<infinilm::runtime::Tensor> &attention_mask,
                                               backends::AttentionBackend attention_backend) const {
    runtime.assertOwnerThread();
    auto residual = hidden_states;
    auto x = layer_norm1_->forward(runtime, hidden_states);
    x = self_attn_->forward(runtime, x, attention_mask, attention_backend);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), x, residual, ops_output);
        INFINILM_TRACE_OP(Add, ops_output);
        x = std::move(ops_output);
    }

    residual = x;
    x = layer_norm2_->forward(runtime, x);
    x = mlp_->forward(runtime, x);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), x, residual, ops_output);
        INFINILM_TRACE_OP(Add, ops_output);
        x = std::move(ops_output);
    }
    return x;
}

SiglipEncoder::SiglipEncoder(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                             const infinilm::runtime::DataType &dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("SiglipEncoder construction requires eager Runtime");
    try {
    size_t num_layers = config.value("num_hidden_layers", 12);
    layers_ = [&] {
        std::vector<std::shared_ptr<SiglipEncoderLayer>> assembled;
        assembled.reserve(num_layers);
        for (size_t i = 0; i < num_layers; ++i)
            assembled.push_back(this->bind_child("layers." + std::to_string(i), std::make_shared<SiglipEncoderLayer>(runtime, config, dtype)));
        return assembled;
    }();

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor SiglipEncoder::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
                                          const std::optional<infinilm::runtime::Tensor> &attention_mask,
                                          backends::AttentionBackend attention_backend) const {
    runtime.assertOwnerThread();
    auto x = hidden_states;
    for (const auto &layer : layers_) {
        x = layer->forward(runtime, x, attention_mask, attention_backend);
    }
    return x;
}

SiglipVisionModel::SiglipVisionModel(infinilm::runtime::Runtime &runtime, const nlohmann::json &config,
                                     const infinilm::runtime::DataType &dtype,
                                     bool drop_last_layer)
    : config_(config), drop_last_layer_(drop_last_layer) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("SiglipVisionModel construction requires eager Runtime");
    try {
    embeddings_ = this->bind_child("embeddings", std::make_shared<std::remove_reference_t<decltype(*embeddings_ )>>(runtime, config, dtype));
    encoder_ = this->bind_child("encoder", std::make_shared<std::remove_reference_t<decltype(*encoder_ )>>(runtime, config, dtype));
    size_t hidden_size = config.value("hidden_size", 768);
    float layer_norm_eps = config.value("layer_norm_eps", 1e-6f);
    post_layernorm_ = this->bind_child("post_layernorm", std::make_shared<std::remove_reference_t<decltype(*post_layernorm_ )>>(runtime, hidden_size, layer_norm_eps, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor SiglipVisionModel::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &pixel_values,
                                              const infinilm::runtime::Tensor &tgt_sizes,
                                              backends::AttentionBackend attention_backend) const {
    runtime.assertOwnerThread();
    auto hidden_states = embeddings_->forward(runtime, pixel_values, tgt_sizes);
    hidden_states = encoder_->forward(runtime, hidden_states, std::nullopt, attention_backend);
    return post_layernorm_->forward(runtime, hidden_states);
}

} // namespace infinilm::models::minicpmv
