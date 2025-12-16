#include "minicpmv_impl.hpp"

#include "infinicore_infer.h"
#include "ref_ops.hpp"
#include "ref_pos_embed.hpp"

#include "../inference_context.hpp"
#include "../../cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

__C struct MiniCPMVModel *
createMiniCPMVModel(const MiniCPMVMeta *meta,
                    const MiniCPMVWeights *weights,
                    infiniDevice_t device,
                    int ndev,
                    const int *dev_ids) {
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    return new MiniCPMVModel(meta, weights, device, std::move(device_ids));
}

__C void destroyMiniCPMVModel(struct MiniCPMVModel *model) { delete model; }

static std::shared_ptr<Tensor> make_host_pos_embed(infiniDtype_t dtype,
                                                   size_t embed_dim,
                                                   uint32_t tgt_h,
                                                   uint32_t tgt_w) {
    auto pos_f32 = minicpmv::ref_pos_embed::make_2d_sincos_pos_embed(embed_dim, tgt_h, tgt_w);
    const size_t n = static_cast<size_t>(tgt_h) * static_cast<size_t>(tgt_w);

    if (dtype == INFINI_DTYPE_F32) {
        return Tensor::weight(pos_f32.data(), INFINI_DTYPE_F32, {n, embed_dim});
    }

    std::vector<uint16_t> packed(n * embed_dim);
    for (size_t i = 0; i < n * embed_dim; ++i) {
        packed[i] = (dtype == INFINI_DTYPE_BF16) ? f32_to_bf16(pos_f32[i]) : f32_to_f16(pos_f32[i]);
    }
    return Tensor::weight(packed.data(), dtype, {n, embed_dim});
}

static uint32_t bucketize_pos(uint32_t idx, uint32_t nb_patches, uint32_t num_patches_per_side) {
    // Equivalent to torch.bucketize(i/nb, boundaries=arange(1/N..), right=True)
    // boundaries are uniform, so this becomes floor(i * N / nb).
    return (idx * num_patches_per_side) / nb_patches;
}

__C void inferMiniCPMVSiglipEmbeddings(struct MiniCPMVModel *model,
                                      const void *pixel_values,
                                      size_t seq_len,
                                      uint32_t tgt_h,
                                      uint32_t tgt_w,
                                      void *output) {
    ASSERT_VALID_PTR(model);
    ASSERT_VALID_PTR(pixel_values);
    ASSERT_VALID_PTR(output);
    ASSERT_EQ(model->device, INFINI_DEVICE_CPU);
    ASSERT_EQ(model->dev_ids.size(), size_t(1));
    RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[0]));

    const auto &vm = model->meta.vision_meta;
    const auto dt = model->meta.language_meta.dt_logits;
    ASSERT_EQ(seq_len, static_cast<size_t>(tgt_h) * static_cast<size_t>(tgt_w));
    ASSERT_EQ(vm.patch_size, size_t(14));
    ASSERT_EQ(vm.vision_num_positions, size_t(4900));

    ASSERT_VALID_PTR(model->weights);
    ASSERT_VALID_PTR(model->weights->vpm_patch_embedding_weight);
    ASSERT_VALID_PTR(model->weights->vpm_patch_embedding_bias);
    ASSERT_VALID_PTR(model->weights->vpm_position_embedding);

    CacheManager cache_manager(100);
    auto memory_pool = std::make_shared<MemoryPool>(256 * 1024 * 1024);
    infiniopHandle_t handle = nullptr;
    infinirtStream_t stream = nullptr;
    RUN_INFINI(infiniopCreateHandle(&handle));
    RUN_INFINI(infinirtStreamCreate(&stream));
    InferenceContext ctx(handle, memory_pool, &cache_manager, stream);
    setInferenceContext(&ctx);

    // Input: [1, 3, patch, seq_len*patch]
    auto x = Tensor::weight(const_cast<void *>(pixel_values), dt,
                            {1, 3, vm.patch_size, seq_len * vm.patch_size});
    auto w = Tensor::weight(const_cast<void *>(model->weights->vpm_patch_embedding_weight),
                            dt, {vm.vision_embed_dim, 3, vm.patch_size, vm.patch_size});
    auto b = Tensor::weight(const_cast<void *>(model->weights->vpm_patch_embedding_bias),
                            dt, {vm.vision_embed_dim});

    auto y = Tensor::buffer(dt, {1, vm.vision_embed_dim, 1, seq_len}, memory_pool);
    std::vector<size_t> pads{0, 0};
    std::vector<size_t> strides{vm.patch_size, vm.patch_size};
    std::vector<size_t> dilations{1, 1};
    conv2d(y, x, w, b, pads, strides, dilations);
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // Create output [seq_len, embed_dim] and copy from NCHW conv output.
    auto out = Tensor::buffer(dt, {seq_len, vm.vision_embed_dim}, memory_pool);
    const size_t unit = dsize(dt);
    const char *y_ptr = reinterpret_cast<const char *>(y->data());
    char *out_ptr = reinterpret_cast<char *>(out->data());

    // y layout: [1, C, 1, W] contiguous => index = c*W + w.
    for (size_t widx = 0; widx < seq_len; ++widx) {
        for (size_t c = 0; c < vm.vision_embed_dim; ++c) {
            const float v = minicpmv::ref_ops::read_as_f32(y_ptr + (c * seq_len + widx) * unit, dt);
            minicpmv::ref_ops::write_from_f32(out_ptr + (widx * vm.vision_embed_dim + c) * unit, dt, v);
        }
    }

    // Add position embedding: position_embedding[position_id] where position_id is computed via bucketing.
    auto pos_table = Tensor::weight(const_cast<void *>(model->weights->vpm_position_embedding),
                                    dt, {vm.vision_num_positions, vm.vision_embed_dim});
    const char *p_ptr = reinterpret_cast<const char *>(pos_table->data());

    const uint32_t N = static_cast<uint32_t>(vm.vision_image_size / vm.patch_size); // 70
    for (uint32_t ih = 0; ih < tgt_h; ++ih) {
        const uint32_t bh = bucketize_pos(ih, tgt_h, N);
        for (uint32_t iw = 0; iw < tgt_w; ++iw) {
            const uint32_t bw = bucketize_pos(iw, tgt_w, N);
            const uint32_t pos_id = bh * N + bw; // [0, 4899]
            const size_t row = static_cast<size_t>(ih) * static_cast<size_t>(tgt_w) + iw;
            for (size_t c = 0; c < vm.vision_embed_dim; ++c) {
                const float base = minicpmv::ref_ops::read_as_f32(out_ptr + (row * vm.vision_embed_dim + c) * unit, dt);
                const float pe = minicpmv::ref_ops::read_as_f32(p_ptr + (static_cast<size_t>(pos_id) * vm.vision_embed_dim + c) * unit, dt);
                minicpmv::ref_ops::write_from_f32(out_ptr + (row * vm.vision_embed_dim + c) * unit, dt, base + pe);
            }
        }
    }

    RUN_INFINI(infinirtMemcpy(output, out->data(), out->numel() * unit, INFINIRT_MEMCPY_D2H));

    setInferenceContext(nullptr);
    infinirtStreamDestroy(stream);
    infiniopDestroyHandle(handle);
}

__C void inferMiniCPMVSiglipLayer(struct MiniCPMVModel *model,
                                 uint32_t layer_idx,
                                 const void *hidden_states,
                                 size_t seq_len,
                                 void *output) {
    ASSERT_VALID_PTR(model);
    ASSERT_VALID_PTR(hidden_states);
    ASSERT_VALID_PTR(output);
    ASSERT_EQ(model->device, INFINI_DEVICE_CPU);
    ASSERT_EQ(model->dev_ids.size(), size_t(1));
    RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[0]));

    const auto &vm = model->meta.vision_meta;
    const auto dt = model->meta.language_meta.dt_logits;
    ASSERT_EQ(vm.vision_embed_dim % vm.vision_num_heads, size_t(0));
    ASSERT(layer_idx < vm.vision_num_layers);
    const size_t nh = vm.vision_num_heads;
    const size_t d = vm.vision_embed_dim;
    const size_t dh = d / nh;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    ASSERT_VALID_PTR(model->weights);
    ASSERT_VALID_PTR(model->weights->vpm_layers);
    const MiniCPMVSiglipLayerWeights *lw = &model->weights->vpm_layers[layer_idx];

    CacheManager cache_manager(100);
    auto memory_pool = std::make_shared<MemoryPool>(512 * 1024 * 1024);
    infiniopHandle_t handle = nullptr;
    infinirtStream_t stream = nullptr;
    RUN_INFINI(infiniopCreateHandle(&handle));
    RUN_INFINI(infinirtStreamCreate(&stream));
    InferenceContext ctx(handle, memory_pool, &cache_manager, stream);
    setInferenceContext(&ctx);

    auto x = Tensor::weight(const_cast<void *>(hidden_states), dt, {seq_len, d});
    auto y = Tensor::buffer(dt, {seq_len, d}, memory_pool);
    rearrange(y, x);
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // LN1
    auto ln1_w = Tensor::weight(const_cast<void *>(lw->layer_norm1_weight), dt, {d});
    auto ln1_b = Tensor::weight(const_cast<void *>(lw->layer_norm1_bias), dt, {d});
    auto x_ln1 = Tensor::buffer(dt, {seq_len, d}, memory_pool);
    minicpmv::ref_ops::layer_norm_last_dim(x_ln1, y, ln1_w, ln1_b, model->meta.vision_meta.vision_layer_norm_eps);

    // Q/K/V projections. We expect weights are pre-transposed to [in_dim, out_dim] for GEMM.
    auto wq = Tensor::weight(const_cast<void *>(lw->q_weight), dt, {d, d});
    auto bq = Tensor::weight(const_cast<void *>(lw->q_bias), dt, {d});
    auto wk = Tensor::weight(const_cast<void *>(lw->k_weight), dt, {d, d});
    auto bk = Tensor::weight(const_cast<void *>(lw->k_bias), dt, {d});
    auto wv = Tensor::weight(const_cast<void *>(lw->v_weight), dt, {d, d});
    auto bv = Tensor::weight(const_cast<void *>(lw->v_bias), dt, {d});

    auto q = Tensor::buffer(dt, {seq_len, d}, memory_pool);
    auto k = Tensor::buffer(dt, {seq_len, d}, memory_pool);
    auto v = Tensor::buffer(dt, {seq_len, d}, memory_pool);
    linear(q, x_ln1, wq, 1.0f, 0.0f, nullptr, bq);
    linear(k, x_ln1, wk, 1.0f, 0.0f, nullptr, bk);
    linear(v, x_ln1, wv, 1.0f, 0.0f, nullptr, bv);
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // Attention per-head using slices to avoid non-contiguous view issues.
    auto attn_out = Tensor::buffer(dt, {seq_len, d}, memory_pool);
    auto scores = Tensor::buffer(dt, {seq_len, seq_len}, memory_pool);
    auto out_h = Tensor::buffer(dt, {seq_len, dh}, memory_pool);
    auto q_h_contig = Tensor::buffer(dt, {seq_len, dh}, memory_pool);
    auto k_h_contig = Tensor::buffer(dt, {seq_len, dh}, memory_pool);
    auto v_h_contig = Tensor::buffer(dt, {seq_len, dh}, memory_pool);
    auto k_t_contig = Tensor::buffer(dt, {dh, seq_len}, memory_pool);

    for (size_t h = 0; h < nh; ++h) {
        const size_t col = h * dh;
        auto q_h = q->slice(1, col, dh); // [L, dh] (strided view)
        auto k_h = k->slice(1, col, dh); // [L, dh] (strided view)
        auto v_h = v->slice(1, col, dh); // [L, dh] (strided view)
        rearrange(q_h_contig, q_h);
        rearrange(k_h_contig, k_h);
        rearrange(v_h_contig, v_h);
        auto k_t_view = k_h_contig->permute({1, 0}); // [dh, L] (strided view)
        rearrange(k_t_contig, k_t_view);             // materialize to contiguous for GEMM

        gemm(scores, q_h_contig, k_t_contig, scale, 0.0f);  // [L, L]
        RUN_INFINI(infinirtStreamSynchronize(stream));
        minicpmv::ref_ops::softmax_last_dim_inplace(scores);
        gemm(out_h, scores, v_h_contig, 1.0f, 0.0f); // [L, dh]
        rearrange(attn_out->slice(1, col, dh), out_h);
    }
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // Out proj + residual
    auto wo = Tensor::weight(const_cast<void *>(lw->out_weight), dt, {d, d});
    auto bo = Tensor::weight(const_cast<void *>(lw->out_bias), dt, {d});
    auto attn_proj = Tensor::buffer(dt, {seq_len, d}, memory_pool);
    linear(attn_proj, attn_out, wo, 1.0f, 0.0f, nullptr, bo);
    add(attn_proj, attn_proj, y);
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // LN2
    auto ln2_w = Tensor::weight(const_cast<void *>(lw->layer_norm2_weight), dt, {d});
    auto ln2_b = Tensor::weight(const_cast<void *>(lw->layer_norm2_bias), dt, {d});
    auto x_ln2 = Tensor::buffer(dt, {seq_len, d}, memory_pool);
    minicpmv::ref_ops::layer_norm_last_dim(x_ln2, attn_proj, ln2_w, ln2_b, model->meta.vision_meta.vision_layer_norm_eps);

    // MLP: fc1 -> gelu_tanh -> fc2 -> residual
    auto w1 = Tensor::weight(const_cast<void *>(lw->fc1_weight), dt, {d, vm.vision_intermediate_size});
    auto b1 = Tensor::weight(const_cast<void *>(lw->fc1_bias), dt, {vm.vision_intermediate_size});
    auto w2 = Tensor::weight(const_cast<void *>(lw->fc2_weight), dt, {vm.vision_intermediate_size, d});
    auto b2 = Tensor::weight(const_cast<void *>(lw->fc2_bias), dt, {d});

    auto fc1 = Tensor::buffer(dt, {seq_len, vm.vision_intermediate_size}, memory_pool);
    linear(fc1, x_ln2, w1, 1.0f, 0.0f, nullptr, b1);
    RUN_INFINI(infinirtStreamSynchronize(stream));
    minicpmv::ref_ops::gelu_tanh_inplace(fc1);

    auto fc2 = Tensor::buffer(dt, {seq_len, d}, memory_pool);
    linear(fc2, fc1, w2, 1.0f, 0.0f, nullptr, b2);
    add(fc2, fc2, attn_proj);
    RUN_INFINI(infinirtStreamSynchronize(stream));

    RUN_INFINI(infinirtMemcpy(output, fc2->data(), fc2->numel() * dsize(dt), INFINIRT_MEMCPY_D2H));

    setInferenceContext(nullptr);
    infinirtStreamDestroy(stream);
    infiniopDestroyHandle(handle);
}

__C void inferMiniCPMVSiglipLayer0(struct MiniCPMVModel *model,
                                  const void *hidden_states,
                                  size_t seq_len,
                                  void *output) {
    inferMiniCPMVSiglipLayer(model, 0, hidden_states, seq_len, output);
}

__C void inferMiniCPMVSiglipEncoder(struct MiniCPMVModel *model,
                                   uint32_t num_layers,
                                   const void *hidden_states,
                                   size_t seq_len,
                                   void *output) {
    ASSERT_VALID_PTR(model);
    ASSERT_VALID_PTR(hidden_states);
    ASSERT_VALID_PTR(output);

    // This step API is for reference CPU-only validation.
    ASSERT_EQ(model->device, INFINI_DEVICE_CPU);
    ASSERT_EQ(model->dev_ids.size(), size_t(1));

    const auto &vm = model->meta.vision_meta;
    const auto dt = model->meta.language_meta.dt_logits;
    ASSERT(num_layers <= vm.vision_num_layers);

    ASSERT_VALID_PTR(model->weights);
    ASSERT_VALID_PTR(model->weights->vpm_post_layernorm_weight);
    ASSERT_VALID_PTR(model->weights->vpm_post_layernorm_bias);

    const size_t d = vm.vision_embed_dim;
    const size_t bytes = seq_len * d * dsize(dt);

    std::vector<uint8_t> buf_a(bytes);
    std::vector<uint8_t> buf_b(bytes);
    std::memcpy(buf_a.data(), hidden_states, bytes);

    uint8_t *buf_in = buf_a.data();
    uint8_t *buf_out = buf_b.data();
    for (uint32_t i = 0; i < num_layers; ++i) {
        inferMiniCPMVSiglipLayer(model, i, buf_in, seq_len, buf_out);
        std::swap(buf_in, buf_out);
    }

    // Apply post-layernorm.
    minicpmv::ref_ops::layer_norm_last_dim_raw(
        output,
        buf_in,
        model->weights->vpm_post_layernorm_weight,
        model->weights->vpm_post_layernorm_bias,
        dt,
        seq_len,
        d,
        vm.vision_layer_norm_eps);
}

__C void inferMiniCPMVResampler(struct MiniCPMVModel *model,
                               const void *x,
                               size_t seq_len,
                               uint32_t tgt_h,
                               uint32_t tgt_w,
                               void *output) {
    ASSERT_VALID_PTR(model);
    ASSERT_VALID_PTR(x);
    ASSERT_VALID_PTR(output);

    // This step API is for reference CPU-only validation.
    ASSERT_EQ(model->device, INFINI_DEVICE_CPU);
    ASSERT_EQ(model->dev_ids.size(), size_t(1));
    RUN_INFINI(infinirtSetDevice(model->device, model->dev_ids[0]));

    const auto &rm = model->meta.resampler_meta;
    const auto dt = model->meta.language_meta.dt_logits;
    ASSERT_EQ(rm.embed_dim, model->meta.language_meta.d);
    ASSERT_EQ(rm.num_heads, model->meta.language_meta.nh);
    ASSERT_EQ(seq_len, static_cast<size_t>(tgt_h) * static_cast<size_t>(tgt_w));

    CacheManager cache_manager(100);
    auto memory_pool = std::make_shared<MemoryPool>(256 * 1024 * 1024);
    infiniopHandle_t handle = nullptr;
    infinirtStream_t stream = nullptr;
    RUN_INFINI(infiniopCreateHandle(&handle));
    RUN_INFINI(infinirtStreamCreate(&stream));
    InferenceContext ctx(handle, memory_pool, &cache_manager, stream);
    setInferenceContext(&ctx);

    // Load inputs/weights into Tensor objects on CPU "device" memory.
    auto x_in = Tensor::weight(const_cast<void *>(x), dt, {seq_len, rm.kv_dim});
    auto kv_w = Tensor::weight(const_cast<void *>(model->weights->resampler_kv_proj_weight), dt, {rm.kv_dim, rm.embed_dim});
    auto q_param = Tensor::weight(const_cast<void *>(model->weights->resampler_query), dt, {rm.num_queries, rm.embed_dim});

    auto ln_q_w = Tensor::weight(const_cast<void *>(model->weights->resampler_ln_q_weight), dt, {rm.embed_dim});
    auto ln_q_b = Tensor::weight(const_cast<void *>(model->weights->resampler_ln_q_bias), dt, {rm.embed_dim});
    auto ln_kv_w = Tensor::weight(const_cast<void *>(model->weights->resampler_ln_kv_weight), dt, {rm.embed_dim});
    auto ln_kv_b = Tensor::weight(const_cast<void *>(model->weights->resampler_ln_kv_bias), dt, {rm.embed_dim});
    auto ln_post_w = Tensor::weight(const_cast<void *>(model->weights->resampler_ln_post_weight), dt, {rm.embed_dim});
    auto ln_post_b = Tensor::weight(const_cast<void *>(model->weights->resampler_ln_post_bias), dt, {rm.embed_dim});

    // x_proj = x_in @ kv_w
    auto x_proj = Tensor::buffer(dt, {seq_len, rm.embed_dim}, memory_pool);
    gemm(x_proj, x_in, kv_w, 1.0f, 0.0f);
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // ln_kv(x_proj)
    minicpmv::ref_ops::layer_norm_last_dim(x_proj, x_proj, ln_kv_w, ln_kv_b, rm.layer_norm_eps);

    // In the reference implementation, pos_embed is added to KEY only:
    //   attn(q, key=x+pos, value=x)
    auto x_val = x_proj;
    auto x_key = Tensor::buffer(dt, {seq_len, rm.embed_dim}, memory_pool);
    rearrange(x_key, x_val);
    auto pos = make_host_pos_embed(dt, rm.embed_dim, tgt_h, tgt_w);
    add(x_key, x_key, pos);
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // q = ln_q(query)
    auto q_ln = Tensor::buffer(dt, {rm.num_queries, rm.embed_dim}, memory_pool);
    rearrange(q_ln, q_param);
    RUN_INFINI(infinirtStreamSynchronize(stream));
    minicpmv::ref_ops::layer_norm_last_dim(q_ln, q_ln, ln_q_w, ln_q_b, rm.layer_norm_eps);

    // In-proj: use pre-transposed weights [D, 3D], then slice into q/k/v.
    auto in_w_full = Tensor::weight(const_cast<void *>(model->weights->resampler_attn_in_proj_weight),
                                    dt, {rm.embed_dim, 3 * rm.embed_dim});
    auto in_b_full = Tensor::weight(const_cast<void *>(model->weights->resampler_attn_in_proj_bias),
                                    dt, {3 * rm.embed_dim});

    auto w_q = in_w_full->slice(1, 0, rm.embed_dim);
    auto w_k = in_w_full->slice(1, rm.embed_dim, rm.embed_dim);
    auto w_v = in_w_full->slice(1, 2 * rm.embed_dim, rm.embed_dim);
    auto b_q = in_b_full->slice(0, 0, rm.embed_dim);
    auto b_k = in_b_full->slice(0, rm.embed_dim, rm.embed_dim);
    auto b_v = in_b_full->slice(0, 2 * rm.embed_dim, rm.embed_dim);

    auto q_full = Tensor::buffer(dt, {rm.num_queries, rm.embed_dim}, memory_pool);
    auto k_full = Tensor::buffer(dt, {seq_len, rm.embed_dim}, memory_pool);
    auto v_full = Tensor::buffer(dt, {seq_len, rm.embed_dim}, memory_pool);
    linear(q_full, q_ln, w_q, 1.0f, 0.0f, nullptr, b_q);
    linear(k_full, x_key, w_k, 1.0f, 0.0f, nullptr, b_k);
    linear(v_full, x_val, w_v, 1.0f, 0.0f, nullptr, b_v);
    RUN_INFINI(infinirtStreamSynchronize(stream));

    const size_t nh = rm.num_heads;
    const size_t dh = rm.embed_dim / nh;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    // Avoid permute+view on non-contiguous tensors: slice heads from the last dim directly.
    auto out_merge = Tensor::buffer(dt, {rm.num_queries, rm.embed_dim}, memory_pool);
    auto qk = Tensor::buffer(dt, {rm.num_queries, seq_len}, memory_pool);
    auto out_h = Tensor::buffer(dt, {rm.num_queries, dh}, memory_pool);
    auto q_h_contig = Tensor::buffer(dt, {rm.num_queries, dh}, memory_pool);
    auto k_h_contig = Tensor::buffer(dt, {seq_len, dh}, memory_pool);
    auto v_h_contig = Tensor::buffer(dt, {seq_len, dh}, memory_pool);
    auto k_t_contig = Tensor::buffer(dt, {dh, seq_len}, memory_pool);

    for (size_t h = 0; h < nh; ++h) {
        const size_t col = h * dh;
        auto q_h = q_full->slice(1, col, dh); // [Q, dh] (strided view)
        auto k_h = k_full->slice(1, col, dh); // [L, dh] (strided view)
        auto v_h = v_full->slice(1, col, dh); // [L, dh] (strided view)
        rearrange(q_h_contig, q_h);
        rearrange(k_h_contig, k_h);
        rearrange(v_h_contig, v_h);
        auto k_h_t_view = k_h_contig->permute({1, 0}); // [dh, L] (strided view)
        rearrange(k_t_contig, k_h_t_view);             // materialize to contiguous for GEMM

        gemm(qk, q_h_contig, k_t_contig, scale, 0.0f);
        RUN_INFINI(infinirtStreamSynchronize(stream));
        minicpmv::ref_ops::softmax_last_dim_inplace(qk);

        gemm(out_h, qk, v_h_contig, 1.0f, 0.0f);
        rearrange(out_merge->slice(1, col, dh), out_h);
    }
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // Out proj: assume already transposed to [embed_dim, embed_dim]
    auto out_w = Tensor::weight(const_cast<void *>(model->weights->resampler_attn_out_proj_weight), dt, {rm.embed_dim, rm.embed_dim});
    auto out_b = Tensor::weight(const_cast<void *>(model->weights->resampler_attn_out_proj_bias), dt, {rm.embed_dim});
    auto out_proj = Tensor::buffer(dt, {rm.num_queries, rm.embed_dim}, memory_pool);
    linear(out_proj, out_merge, out_w, 1.0f, 0.0f, nullptr, out_b);
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // ln_post and final projection
    minicpmv::ref_ops::layer_norm_last_dim(out_proj, out_proj, ln_post_w, ln_post_b, rm.layer_norm_eps);
    auto proj_w = Tensor::weight(const_cast<void *>(model->weights->resampler_proj), dt, {rm.embed_dim, rm.embed_dim});
    auto out_final = Tensor::buffer(dt, {rm.num_queries, rm.embed_dim}, memory_pool);
    gemm(out_final, out_proj, proj_w, 1.0f, 0.0f);
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // Copy back to caller.
    RUN_INFINI(infinirtMemcpy(output, out_final->data(),
                              out_final->numel() * dsize(dt),
                              INFINIRT_MEMCPY_D2H));

    setInferenceContext(nullptr);
    infinirtStreamDestroy(stream);
    infiniopDestroyHandle(handle);
}

__C void inferMiniCPMVVisionResampler(struct MiniCPMVModel *model,
                                     const void *pixel_values,
                                     size_t seq_len,
                                     uint32_t tgt_h,
                                     uint32_t tgt_w,
                                     void *output) {
    ASSERT_VALID_PTR(model);
    ASSERT_VALID_PTR(pixel_values);
    ASSERT_VALID_PTR(output);

    // This step API is for reference CPU-only validation.
    ASSERT_EQ(model->device, INFINI_DEVICE_CPU);
    ASSERT_EQ(model->dev_ids.size(), size_t(1));

    const auto &vm = model->meta.vision_meta;
    const auto dt = model->meta.language_meta.dt_logits;
    ASSERT_EQ(seq_len, static_cast<size_t>(tgt_h) * static_cast<size_t>(tgt_w));

    const size_t d = vm.vision_embed_dim;
    const size_t bytes = seq_len * d * dsize(dt);
    std::vector<uint8_t> buf(bytes);

    inferMiniCPMVSiglipEmbeddings(model, pixel_values, seq_len, tgt_h, tgt_w, buf.data());
    inferMiniCPMVSiglipEncoder(model, static_cast<uint32_t>(vm.vision_num_layers), buf.data(), seq_len, buf.data());
    inferMiniCPMVResampler(model, buf.data(), seq_len, tgt_h, tgt_w, output);
}
