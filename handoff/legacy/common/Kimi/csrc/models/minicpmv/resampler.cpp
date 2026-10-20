#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/linear/dense.hpp"
#include "resampler.hpp"

#include "../../utils.hpp"

#include "runtime/context.hpp"
#include "runtime/runtime.hpp"
#include <memory>
#include <exception>

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace infinilm::models::minicpmv {
namespace {
void compute_2d_sincos_pos_embed(float *out, size_t embed_dim, size_t h, size_t w) {
    const size_t half = embed_dim / 2;
    const size_t quarter = half / 2;

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            float *dst = out + (y * w + x) * embed_dim;
            for (size_t i = 0; i < quarter; ++i) {
                const float omega = std::pow(10000.0f, -static_cast<float>(i) / static_cast<float>(quarter));
                const float a = static_cast<float>(x) * omega;
                const float b = static_cast<float>(y) * omega;
                dst[i] = std::sin(a);
                dst[i + quarter] = std::cos(a);
                dst[i + half] = std::sin(b);
                dst[i + half + quarter] = std::cos(b);
            }
        }
    }
}

void write_pos_embed(void *dst, infinilm::runtime::DataType dtype, const float *src, size_t n) {
    if (dtype == infinilm::runtime::DataType::kFloat32) {
        if (n != 0) std::memcpy(dst, src, n * sizeof(float));
        return;
    }
    if (dtype == infinilm::runtime::DataType::kFloat16) {
        auto *out = reinterpret_cast<uint16_t *>(dst);
        for (size_t i = 0; i < n; ++i) {
            out[i] = f32_to_f16(src[i]);
        }
        return;
    }
    if (dtype == infinilm::runtime::DataType::kBFloat16) {
        auto *out = reinterpret_cast<uint16_t *>(dst);
        for (size_t i = 0; i < n; ++i) {
            out[i] = f32_to_bf16(src[i]);
        }
        return;
    }
    throw std::runtime_error("Unsupported dtype in write_pos_embed");
}

infinilm::runtime::Tensor build_2d_sincos_pos_embed_cpu(size_t h,
                                                 size_t w,
                                                 size_t embed_dim,
                                                 infinilm::runtime::DataType dtype) {
    std::vector<float> buf(h * w * embed_dim);
    compute_2d_sincos_pos_embed(buf.data(), embed_dim, h, w);
    // Owned, zero-initialized CPU bytes; no implicit Runtime/zeros submission.
    auto owner = std::make_shared<std::vector<unsigned char>>(buf.size() * infinilm::runtime::dsize(dtype), 0);
    auto embedding_cpu = infinilm::runtime::Tensor::from_blob(owner->data(),
        {h, w, embed_dim}, dtype, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu}, owner);
    write_pos_embed(embedding_cpu->data(), embedding_cpu->dtype(), buf.data(), buf.size());
    return embedding_cpu;
}
} // namespace

ResamplerAttention::ResamplerAttention(infinilm::runtime::Runtime &runtime, size_t embed_dim,
                                       size_t num_heads,
                                       const infinilm::runtime::DataType &dtype)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(num_heads ? embed_dim / num_heads : 0),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("ResamplerAttention construction requires eager Runtime");
    try {
    if (embed_dim_ == 0 || num_heads_ == 0 || embed_dim_ % num_heads_ != 0) {
        throw std::runtime_error("ResamplerAttention: embed_dim must be divisible by num_heads");
    }
    in_proj_weight_ = infinilm::weights::Weight(runtime, {3 * embed_dim_, embed_dim_}, dtype); this->bind_weight("in_proj_weight", in_proj_weight_);
    in_proj_bias_ = infinilm::weights::Weight(runtime, {3 * embed_dim_}, dtype); this->bind_weight("in_proj_bias", in_proj_bias_);
    out_proj_ = this->bind_child("out_proj", std::make_shared<std::remove_reference_t<decltype(*out_proj_ )>>(runtime, embed_dim_, embed_dim_, true, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor ResamplerAttention::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &query,
                                               const infinilm::runtime::Tensor &key,
                                               const infinilm::runtime::Tensor &value) const {
    runtime.assertOwnerThread();
    auto batch_size = query->size(0);
    auto q_len = query->size(1);
    auto k_len = key->size(1);

    auto Wq = in_proj_weight_->narrow({{0, 0, embed_dim_}});
    auto Wk = in_proj_weight_->narrow({{0, embed_dim_, embed_dim_}});
    auto Wv = in_proj_weight_->narrow({{0, 2 * embed_dim_, embed_dim_}});
    auto bq = in_proj_bias_->narrow({{0, 0, embed_dim_}});
    auto bk = in_proj_bias_->narrow({{0, embed_dim_, embed_dim_}});
    auto bv = in_proj_bias_->narrow({{0, 2 * embed_dim_, embed_dim_}});

    auto q2d = query->view({batch_size * q_len, embed_dim_});
    auto k2d = key->view({batch_size * k_len, embed_dim_});
    auto v2d = value->view({batch_size * k_len, embed_dim_});

    auto q_proj = infinilm::layers::linear::dense_projection(runtime, q2d, Wq, std::make_optional<infinilm::runtime::Tensor>(bq))
                      ->view({batch_size, q_len, embed_dim_});
    auto k_proj = infinilm::layers::linear::dense_projection(runtime, k2d, Wk, std::make_optional<infinilm::runtime::Tensor>(bk))
                      ->view({batch_size, k_len, embed_dim_});
    auto v_proj = infinilm::layers::linear::dense_projection(runtime, v2d, Wv, std::make_optional<infinilm::runtime::Tensor>(bv))
                      ->view({batch_size, k_len, embed_dim_});

    auto q_reshaped = q_proj->view({batch_size, q_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous(runtime);
    auto k_reshaped = k_proj->view({batch_size, k_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous(runtime);
    auto v_reshaped = v_proj->view({batch_size, k_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous(runtime);

    auto q_flat = q_reshaped->view({batch_size * num_heads_, q_len, head_dim_});
    auto k_flat = k_reshaped->view({batch_size * num_heads_, k_len, head_dim_});
    auto v_flat = v_reshaped->view({batch_size * num_heads_, k_len, head_dim_});

    auto k_t = k_flat->permute({0, 2, 1});
    auto attn_weights_shape = q_flat->shape();
    attn_weights_shape.back() = k_t->size(q_flat->ndim() - 1);
    auto attn_weights = infinilm::runtime::Tensor::empty(runtime, attn_weights_shape, q_flat->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Gemm::DefaultConfig(runtime.device().type()), q_flat, k_t,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{scale_},
        std::optional<float>{0.0f}, std::optional<int>{}, std::optional<int>{}, attn_weights);
    INFINILM_TRACE_OP(Gemm, attn_weights);
    auto attn_view = attn_weights->view({batch_size * num_heads_, q_len, k_len});
    INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
        && runtime.device().type() != infinilm::runtime::Device::Type::kMars
        && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
    infinilm::execution::submit<infini::ops::Softmax>(
        runtime, infini::ops::Softmax::DefaultConfig(runtime.device().type()), attn_view, std::int64_t{-1}, std::optional<infini::ops::DataType>{}, attn_view);

    auto attn_output_shape = attn_weights->shape();
    attn_output_shape.back() = v_flat->size(attn_weights->ndim() - 1);
    auto attn_output = infinilm::runtime::Tensor::empty(runtime, attn_output_shape, attn_weights->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Gemm::DefaultConfig(runtime.device().type()), attn_weights, v_flat,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0f},
        std::optional<float>{0.0f}, std::optional<int>{}, std::optional<int>{}, attn_output);
    INFINILM_TRACE_OP(Gemm, attn_output);
    auto out = attn_output->view({batch_size, num_heads_, q_len, head_dim_})
                   ->permute({0, 2, 1, 3})
                   ->contiguous(runtime)
                   ->view({batch_size, q_len, embed_dim_});

    auto out2d = out->view({batch_size * q_len, embed_dim_});
    auto out_proj = out_proj_->forward(runtime, out2d)->view({batch_size, q_len, embed_dim_});
    return out_proj;
}

Resampler::Resampler(infinilm::runtime::Runtime &runtime, size_t num_queries,
                     size_t embed_dim,
                     size_t num_heads,
                     size_t kv_dim,
                     size_t image_size,
                     size_t patch_size,
                     const infinilm::runtime::DataType &dtype)
    : num_queries_(num_queries),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      kv_dim_(kv_dim),
      image_size_(image_size),
      patch_size_(patch_size),
      use_kv_proj_(kv_dim != embed_dim) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Resampler construction requires eager Runtime");
    if (patch_size_ == 0 || num_heads_ == 0 || embed_dim_ % num_heads_ != 0)
        throw std::invalid_argument("Resampler requires valid patch/head geometry");
    infinilm::runtime::Tensor embedding_table_init;
    try {
    query_ = infinilm::weights::Weight(runtime, {num_queries_, embed_dim_}, dtype); this->bind_weight("query", query_);
    proj_ = infinilm::weights::Weight(runtime, {embed_dim_, embed_dim_}, dtype); this->bind_weight("proj", proj_);

    attn_ = this->bind_child("attn", std::make_shared<std::remove_reference_t<decltype(*attn_ )>>(runtime, embed_dim_, num_heads_, dtype));
    ln_q_ = this->bind_child("ln_q", std::make_shared<std::remove_reference_t<decltype(*ln_q_ )>>(runtime, embed_dim_, 1e-6, dtype));
    ln_kv_ = this->bind_child("ln_kv", std::make_shared<std::remove_reference_t<decltype(*ln_kv_ )>>(runtime, embed_dim_, 1e-6, dtype));
    ln_post_ = this->bind_child("ln_post", std::make_shared<std::remove_reference_t<decltype(*ln_post_ )>>(runtime, embed_dim_, 1e-6, dtype));

    if (use_kv_proj_) {
        kv_proj_ = this->bind_child("kv_proj", std::make_shared<std::remove_reference_t<decltype(*kv_proj_ )>>(runtime, kv_dim_, embed_dim_, false, dtype));
    }

    // Initialize full 2d embeddings with max size, calculate on cpu and copy to gpu
    size_t num_patches = image_size_ / patch_size_;
    embedding_table_ = infinilm::runtime::Tensor::empty_parameter(runtime, {num_patches, num_patches, embed_dim_}, dtype);
    embedding_table_init = build_2d_sincos_pos_embed_cpu(num_patches, num_patches, embed_dim_, dtype);
    embedding_table_->copy_from(runtime, embedding_table_init);
    runtime.syncStream(); // The CPU owner remains alive until upload completion.

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor Resampler::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &x,
                                      const infinilm::runtime::Tensor &tgt_sizes) const {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::invalid_argument("Resampler: data-dependent target-size metadata is not recording-qualified");
    }
    if (!x || x->device() != runtime.device() || !tgt_sizes ||
        tgt_sizes->dtype() != infinilm::runtime::DataType::kInt64 || !tgt_sizes->is_contiguous() ||
        tgt_sizes->ndim() != 2 || tgt_sizes->size(0) != x->size(0) || tgt_sizes->size(1) != 2) {
        throw std::invalid_argument("Resampler: invalid input or target-size metadata");
    }
    std::vector<int64_t> target_values(tgt_sizes->numel());
    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu, 0};
    if (tgt_sizes->device() == cpu) std::memcpy(target_values.data(), tgt_sizes->data(), tgt_sizes->nbytes());
    else if (tgt_sizes->device() == runtime.device()) {
        runtime.memcpyD2H(target_values.data(), tgt_sizes->data(), tgt_sizes->nbytes());
        runtime.syncStream();
    } else throw std::invalid_argument("Resampler: metadata belongs to another device");
    for (size_t i = 0; i < target_values.size(); i += 2) {
        if (target_values[i] <= 0 || target_values[i + 1] <= 0 ||
            static_cast<uint64_t>(target_values[i]) > x->size(1) / static_cast<uint64_t>(target_values[i + 1])) {
            throw std::invalid_argument("Resampler: target grid exceeds sequence capacity");
        }
    }
    auto batch_size = x->size(0);
    auto seq_len = x->size(1);

    auto kv = x;
    if (use_kv_proj_) {
        kv = kv_proj_->forward(runtime, const_cast<infinilm::runtime::Tensor &>(kv));
    }
    kv = ln_kv_->forward(runtime, kv);

    // Build positional embeddings on CPU
    const auto *tgt_sizes_ptr = target_values.data();
    auto pos_embeddings = infinilm::runtime::Tensor::empty(runtime, kv->shape(), kv->dtype());
    auto zero_owner = std::make_shared<std::vector<unsigned char>>(pos_embeddings->nbytes(), 0);
    auto zero_host = infinilm::runtime::Tensor::from_blob(zero_owner->data(), kv->shape(), kv->dtype(), cpu, zero_owner);
    pos_embeddings->copy_from(runtime, zero_host);
    runtime.syncStream();

    for (size_t b = 0; b < batch_size; ++b) {

        auto tgt_h = static_cast<size_t>(tgt_sizes_ptr[b * 2]);
        auto tgt_w = static_cast<size_t>(tgt_sizes_ptr[b * 2 + 1]);

        infinilm::runtime::Tensor src_embeddings;
        const size_t table_h = embedding_table_->size(0);
        const size_t table_w = embedding_table_->size(1);
        if (tgt_h <= table_h && tgt_w <= table_w) {
            src_embeddings = embedding_table_->narrow({{0, 0, tgt_h}, {1, 0, tgt_w}});
        } else {
            std::vector<float> values(tgt_h * tgt_w * embed_dim_);
            compute_2d_sincos_pos_embed(values.data(), embed_dim_, tgt_h, tgt_w);
            std::optional<infinilm::runtime::Tensor> host;
            if (kv->dtype() == infinilm::runtime::DataType::kFloat32) {
                auto owner = std::make_shared<std::vector<float>>(values);
                host.emplace(infinilm::runtime::Tensor::from_blob(owner->data(), {tgt_h, tgt_w, embed_dim_}, kv->dtype(), cpu, owner));
            } else {
                auto owner = std::make_shared<std::vector<uint16_t>>(values.size());
                write_pos_embed(owner->data(), kv->dtype(), values.data(), values.size());
                host.emplace(infinilm::runtime::Tensor::from_blob(owner->data(), {tgt_h, tgt_w, embed_dim_}, kv->dtype(), cpu, owner));
            }
            src_embeddings = infinilm::runtime::Tensor::empty(runtime, {tgt_h, tgt_w, embed_dim_}, kv->dtype());
            src_embeddings->copy_from(runtime, *host);
            runtime.syncStream();
        }
        auto tgt_embeddings = pos_embeddings->narrow({{0, b, 1}, {1, 0, tgt_h * tgt_w}})->view({tgt_h, tgt_w, embed_dim_});
        tgt_embeddings->copy_from(runtime, src_embeddings);
    }

    auto kv_with_pos = infinilm::runtime::Tensor::empty(runtime, kv->shape(), kv->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), kv, pos_embeddings, kv_with_pos);
    INFINILM_TRACE_OP(Add, kv_with_pos);

    auto q = ln_q_->forward(runtime, query_);
    if (q->shape().size() == 2) {
        q = q->unsqueeze(0);
    }
    auto q_batched = infinilm::runtime::Tensor::empty(runtime, {batch_size, num_queries_, embed_dim_}, q->dtype());
    for (size_t b = 0; b < batch_size; ++b) {
        q_batched->narrow({{0, b, 1}})->copy_from(runtime, q);
    }

    auto out = attn_->forward(runtime, q_batched, kv_with_pos, kv);
    out = ln_post_->forward(runtime, out);

    auto out2d = out->view({batch_size * num_queries_, embed_dim_});
    auto proj_out_shape = out2d->shape();
    proj_out_shape.back() = proj_.tensor()->size(out2d->ndim() - 1);
    auto proj_out = infinilm::runtime::Tensor::empty(runtime, proj_out_shape, out2d->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Gemm::DefaultConfig(runtime.device().type()), out2d, proj_.tensor(),
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0f},
        std::optional<float>{0.0f}, std::optional<int>{}, std::optional<int>{}, proj_out);
    INFINILM_TRACE_OP(Gemm, proj_out);
    proj_out = proj_out->view({batch_size, num_queries_, embed_dim_});
    return proj_out;
}

} // namespace infinilm::models::minicpmv
