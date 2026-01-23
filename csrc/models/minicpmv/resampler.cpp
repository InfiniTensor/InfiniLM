#include "resampler.hpp"

#include "../../utils.hpp"
#include "infinicore/ops.hpp"

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

void write_pos_embed(void *dst, infinicore::DataType dtype, const float *src, size_t n) {
    if (dtype == infinicore::DataType::F32) {
        std::memcpy(dst, src, n * sizeof(float));
        return;
    }
    if (dtype == infinicore::DataType::F16) {
        auto *out = reinterpret_cast<uint16_t *>(dst);
        for (size_t i = 0; i < n; ++i) out[i] = f32_to_f16(src[i]);
        return;
    }
    if (dtype == infinicore::DataType::BF16) {
        auto *out = reinterpret_cast<uint16_t *>(dst);
        for (size_t i = 0; i < n; ++i) out[i] = f32_to_bf16(src[i]);
        return;
    }
    throw std::runtime_error("Unsupported dtype in write_pos_embed");
}
} // namespace

ResamplerAttention::ResamplerAttention(size_t embed_dim,
                                       size_t num_heads,
                                       const infinicore::DataType &dtype,
                                       const infinicore::Device &device)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    if (embed_dim_ % num_heads_ != 0) {
        throw std::runtime_error("ResamplerAttention: embed_dim must be divisible by num_heads");
    }
    INFINICORE_NN_PARAMETER_INIT(in_proj_weight, ({3 * embed_dim_, embed_dim_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(in_proj_bias, ({3 * embed_dim_}, dtype, device));
    INFINICORE_NN_MODULE_INIT(out_proj, embed_dim_, embed_dim_, true, dtype, device);
}

infinicore::Tensor ResamplerAttention::forward(const infinicore::Tensor &query,
                                               const infinicore::Tensor &key,
                                               const infinicore::Tensor &value) const {
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

    auto q_proj = infinicore::op::linear(q2d, Wq, std::make_optional<infinicore::Tensor>(bq))
                      ->view({batch_size, q_len, embed_dim_});
    auto k_proj = infinicore::op::linear(k2d, Wk, std::make_optional<infinicore::Tensor>(bk))
                      ->view({batch_size, k_len, embed_dim_});
    auto v_proj = infinicore::op::linear(v2d, Wv, std::make_optional<infinicore::Tensor>(bv))
                      ->view({batch_size, k_len, embed_dim_});

    auto q_reshaped = q_proj->view({batch_size, q_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous();
    auto k_reshaped = k_proj->view({batch_size, k_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous();
    auto v_reshaped = v_proj->view({batch_size, k_len, num_heads_, head_dim_})->permute({0, 2, 1, 3})->contiguous();

    auto q_flat = q_reshaped->view({batch_size * num_heads_, q_len, head_dim_});
    auto k_flat = k_reshaped->view({batch_size * num_heads_, k_len, head_dim_});
    auto v_flat = v_reshaped->view({batch_size * num_heads_, k_len, head_dim_});

    auto k_t = k_flat->permute({0, 2, 1});
    auto attn_weights = infinicore::op::matmul(q_flat, k_t, scale_);
    auto attn_view = attn_weights->view({batch_size * num_heads_, q_len, k_len});
    infinicore::op::softmax_(attn_view, attn_view, -1);

    auto attn_output = infinicore::op::matmul(attn_weights, v_flat);
    auto out = attn_output->view({batch_size, num_heads_, q_len, head_dim_})
                   ->permute({0, 2, 1, 3})
                   ->contiguous()
                   ->view({batch_size, q_len, embed_dim_});

    auto out2d = out->view({batch_size * q_len, embed_dim_});
    auto out_proj = out_proj_->forward(out2d)->view({batch_size, q_len, embed_dim_});
    return out_proj;
}

Resampler::Resampler(size_t num_queries,
                     size_t embed_dim,
                     size_t num_heads,
                     size_t kv_dim,
                     const infinicore::DataType &dtype,
                     const infinicore::Device &device)
    : num_queries_(num_queries),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      kv_dim_(kv_dim),
      use_kv_proj_(kv_dim != embed_dim) {
    INFINICORE_NN_PARAMETER_INIT(query, ({num_queries_, embed_dim_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(proj, ({embed_dim_, embed_dim_}, dtype, device));
    INFINICORE_NN_MODULE_INIT(attn, embed_dim_, num_heads_, dtype, device);
    INFINICORE_NN_MODULE_INIT(ln_q, embed_dim_, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(ln_kv, embed_dim_, 1e-6, dtype, device);
    INFINICORE_NN_MODULE_INIT(ln_post, embed_dim_, 1e-6, dtype, device);

    if (use_kv_proj_) {
        INFINICORE_NN_MODULE_INIT(kv_proj, kv_dim_, embed_dim_, false, dtype, device);
    }
}

infinicore::Tensor Resampler::forward(const infinicore::Tensor &x,
                                      const std::optional<infinicore::Tensor> &tgt_sizes) const {
    auto batch_size = x->size(0);
    auto seq_len = x->size(1);

    auto kv = x;
    if (use_kv_proj_) {
        kv = kv_proj_->forward(const_cast<infinicore::Tensor &>(kv));
    }
    kv = ln_kv_->forward(kv);

    // Build positional embeddings on CPU
    std::vector<int64_t> tgt_sizes_host;
    if (tgt_sizes.has_value()) {
        auto tgt_cpu = tgt_sizes.value()->to(infinicore::Device::cpu());
        auto n = tgt_cpu->numel();
        tgt_sizes_host.resize(n);
        std::memcpy(tgt_sizes_host.data(), tgt_cpu->data(), n * sizeof(int64_t));
    }

    auto pos_cpu = infinicore::Tensor::zeros({batch_size, seq_len, embed_dim_}, kv->dtype(), infinicore::Device::cpu());
    auto *pos_ptr = reinterpret_cast<std::byte *>(pos_cpu->data());
    const size_t elem_size = pos_cpu->element_size();

    for (size_t b = 0; b < batch_size; ++b) {
        size_t tgt_h = 1;
        size_t tgt_w = seq_len;
        if (!tgt_sizes_host.empty()) {
            tgt_h = static_cast<size_t>(tgt_sizes_host[b * 2]);
            tgt_w = static_cast<size_t>(tgt_sizes_host[b * 2 + 1]);
        }
        const size_t patch_len = tgt_h * tgt_w;
        std::vector<float> buf(patch_len * embed_dim_);
        compute_2d_sincos_pos_embed(buf.data(), embed_dim_, tgt_h, tgt_w);
        write_pos_embed(pos_ptr + b * seq_len * embed_dim_ * elem_size, pos_cpu->dtype(), buf.data(), patch_len * embed_dim_);
    }

    auto pos = pos_cpu->to(kv->device());
    auto kv_with_pos = infinicore::op::add(kv, pos);

    auto q = ln_q_->forward(query_);
    if (q->shape().size() == 2) {
        q = q->unsqueeze(0);
    }
    auto q_batched = infinicore::Tensor::empty({batch_size, num_queries_, embed_dim_}, q->dtype(), q->device());
    for (size_t b = 0; b < batch_size; ++b) {
        q_batched->narrow({{0, b, 1}})->copy_from(q);
    }

    auto out = attn_->forward(q_batched, kv_with_pos, kv);
    out = ln_post_->forward(out);

    auto out2d = out->view({batch_size * num_queries_, embed_dim_});
    auto proj_out = infinicore::op::matmul(out2d, proj_)->view({batch_size, num_queries_, embed_dim_});
    return proj_out;
}

} // namespace infinilm::models::minicpmv
