#include "kv_compression.hpp"
#include "../utils.hpp"
#include "../tensor.hpp"
#include "../models/inference_context.hpp"
#include "infinicore_infer.h"
#include <infinirt.h>

#include <sstream>

namespace {
// Transpose a 2D weight (out, in) -> (in, out) into a contiguous buffer.
std::shared_ptr<Tensor> make_transposed(std::shared_ptr<Tensor> w, InferenceContext *ctx) {
    if (!w || w->ndim() != 2) return w;
    auto shape = w->shape(); // [out, in]
    auto view_t = w->permute({1, 0}); // view with swapped strides
    auto out = Tensor::buffer(w->dtype(), {shape[1], shape[0]}, ctx->memory_pool);
    out->copyFrom(view_t, ctx->op_handle, ctx->stream);
    return out;
}
} // namespace

std::unique_ptr<CompressedKV> Compressor::compress(const KVCache &kv, uint32_t seq_len) {
    if (!config_.enable) {
        return nullptr;
    }
    if (weights_.empty()) {
        std::cerr << "Compressor::compress: weights are empty" << std::endl;
        return nullptr;
    }
    if (seq_len < config_.min_seq_len) {
        return nullptr;
    }

    auto compressed = std::make_unique<CompressedKV>();
    if (kv.k.empty()) {
        return nullptr;
    }
    const size_t ndev = kv.k.size();
    const size_t nlayers = kv.k[0].size();
    compressed->layers.resize(nlayers);

    // Only handle device 0 for now.
    if (ndev == 0) {
        return nullptr;
    }

    // Validate / auto-initialize inference context for the current device.
    auto ensure_ctx = [&]() -> InferenceContext * {
        auto *ctx_ptr = maybe_get_context();
        if (ctx_ptr && ctx_ptr->op_handle != nullptr && ctx_ptr->memory_pool != nullptr) {
            return ctx_ptr;
        }
        // Auto create a lightweight context bound to the KV device to allow tests to run.
        static CacheManager auto_cache_mgr(32);
        static std::shared_ptr<MemoryPool> auto_pool;
        static infiniopHandle_t auto_handle = nullptr;
        static infinirtStream_t auto_stream = nullptr;
        static InferenceContext *auto_ctx = nullptr;

        if (auto_ctx == nullptr) {
            // Bind to device 0 (first shard) since compressor currently assumes single device.
            auto device_type = kv.k[0][0]->deviceType();
            auto device_id = kv.k[0][0]->deviceId();
            RUN_INFINI(infinirtSetDevice(device_type, device_id));
            auto_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);
            RUN_INFINI(infiniopCreateHandle(&auto_handle));
            RUN_INFINI(infinirtStreamCreate(&auto_stream));
            static InferenceContext ctx(auto_handle, auto_pool, &auto_cache_mgr, auto_stream);
            auto_ctx = &ctx;
        }
        setInferenceContext(auto_ctx);
        return auto_ctx;
    };

    auto *ctx_ptr = ensure_ctx();
    if (!ctx_ptr || ctx_ptr->op_handle == nullptr || ctx_ptr->memory_pool == nullptr) {
        std::cerr << "compress: inference context not initialized (op_handle/memory_pool), fallback to no-compress copy" << std::endl;
        // Fallback: return original KV (device 0) without compression.
        const size_t nlayers = kv.k[0].size();
        auto fallback = std::make_unique<CompressedKV>();
        fallback->layers.resize(nlayers);
        for (size_t layer = 0; layer < nlayers; ++layer) {
            fallback->layers[layer].k_comp = kv.k[0][layer];
            fallback->layers[layer].v_comp = kv.v[0][layer];
            fallback->layers[layer].orig_seq_len = static_cast<uint32_t>(kv.k[0][layer]->shape()[0]);
            fallback->layers[layer].comp_seq_len = fallback->layers[layer].orig_seq_len;
        }
        return fallback;
    }
    config_.compression_factor = 5;
    config_.image_kv_len = 8;
    const uint32_t factor = config_.compression_factor > 0 ? config_.compression_factor : 1;
    std::cout << factor << std::endl;

    for (size_t layer = 0; layer < nlayers; ++layer) {
        auto k_tensor = kv.k[0][layer];
        auto v_tensor = kv.v[0][layer];
        if (!k_tensor || !v_tensor) {
            return nullptr;
        }
        const auto &shape = k_tensor->shape();
        if (shape.size() != 3) {
            return nullptr;
        }
        const uint32_t seq = static_cast<uint32_t>(shape[0]);
        const uint32_t nkvh = static_cast<uint32_t>(shape[1]);
        const uint32_t dk = static_cast<uint32_t>(shape[2]);

        auto fetch = [&](uint32_t prefix, uint32_t slot) -> std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> {
            return getLinearWithBias(static_cast<uint32_t>(layer), prefix, slot);
        };

        auto run_pipeline = [&](std::shared_ptr<Tensor> input2d, uint32_t prefix) -> std::shared_ptr<Tensor> {
            auto l0 = fetch(prefix, 0);
            auto l1 = fetch(prefix, 1);
            auto l2 = fetch(prefix, 2);
            if (!l0.first || !l1.first || !l2.first) {
                return nullptr;
            }
            if (l0.first->shape()[1] != factor * dk ||
                l1.first->shape()[1] != l0.first->shape()[0] ||
                l2.first->shape()[1] != l1.first->shape()[0]) {
                std::cerr << "compress: weight/input shape mismatch at prefix " << prefix
                          << " layer " << layer << std::endl;
                return nullptr;
            }
            auto w0 = make_transposed(l0.first, ctx_ptr);
            auto w1 = make_transposed(l1.first, ctx_ptr);
            auto w2 = make_transposed(l2.first, ctx_ptr);

            // auto w0 = l0.first;
            // auto w1 = l1.first;
            // auto w2 = l2.first;

            const size_t rows_linear = input2d->shape()[0];
            auto out0 = Tensor::buffer(input2d->dtype(), {rows_linear, l0.first->shape()[0]}, ctx_ptr->memory_pool);
            auto out1 = Tensor::buffer(input2d->dtype(), {rows_linear, l1.first->shape()[0]}, ctx_ptr->memory_pool);
            auto out2 = Tensor::buffer(input2d->dtype(), {rows_linear, l2.first->shape()[0]}, ctx_ptr->memory_pool);

            linear(out0, input2d, w0, 1.0f, 0.0f, nullptr, l0.second);
            relu(out0, out0);

            linear(out1, out0, w1, 1.0f, 0.0f, nullptr, l1.second);
            relu(out1, out1);

            linear(out2, out1, w2, 1.0f, 0.0f, nullptr, l2.second);
            return out2;
        };

        auto compress_segment = [&](std::shared_ptr<Tensor> k_seg,
                                    std::shared_ptr<Tensor> v_seg,
                                    uint32_t prefix_base) -> std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> {
            if (!k_seg || !v_seg) return std::make_pair(nullptr, nullptr);
            const auto seg_shape = k_seg->shape();
            uint32_t seg_len = static_cast<uint32_t>(seg_shape[0]);
            uint32_t compressed_seq_len = (seg_len  / factor);
            if (compressed_seq_len < config_.min_seq_len) {
                return {k_seg, v_seg};
            }
            uint32_t compress_len = compressed_seq_len * factor;
            uint32_t remainder_len = seg_len - compress_len;

            auto k_head = k_seg->slice(0, 0, compress_len);
            auto v_head = v_seg->slice(0, 0, compress_len);

            auto k_head_buf = Tensor::buffer(k_seg->dtype(), {compress_len, nkvh, dk}, ctx_ptr->memory_pool);
            k_head_buf->copyFrom(k_head, ctx_ptr->op_handle, ctx_ptr->stream);
            auto v_head_buf = Tensor::buffer(v_seg->dtype(), {compress_len, nkvh, dk}, ctx_ptr->memory_pool);
            v_head_buf->copyFrom(v_head, ctx_ptr->op_handle, ctx_ptr->stream);

            // auto k_grouped = k_head_buf->view({compress_len / factor, nkvh, factor, dk});
            // auto v_grouped = v_head_buf->view({compress_len / factor, nkvh, factor, dk});

            auto k_perm = k_head_buf->permute({1, 0, 2}); // 视图，非连续
            auto k_contig = Tensor::buffer(k_tensor->dtype(), {nkvh, compress_len, dk}, ctx_ptr->memory_pool);
            k_contig->copyFrom(k_perm, ctx_ptr->op_handle, ctx_ptr->stream); 

            auto v_perm = v_head_buf->permute({1, 0, 2});
            auto v_contig = Tensor::buffer(v_tensor->dtype(), {nkvh, compress_len, dk}, ctx_ptr->memory_pool);
            v_contig->copyFrom(v_perm, ctx_ptr->op_handle, ctx_ptr->stream);

            auto k_grouped = k_contig->view({nkvh, compress_len / factor,  factor * dk});
            auto v_grouped = v_contig->view({nkvh, compress_len / factor,  factor * dk});

            const size_t rows_linear = static_cast<size_t>(compress_len / factor) * nkvh;
            auto k_in2d = k_grouped->view({rows_linear, factor * dk});
            auto v_in2d = v_grouped->view({rows_linear, factor * dk});

            auto k_comp2d = run_pipeline(k_in2d, prefix_base);
            auto v_comp2d = run_pipeline(v_in2d, prefix_base + 1);
            if (!k_comp2d || !v_comp2d) {
                return {nullptr, nullptr};
            }

            auto k_comp_head = k_comp2d->view({compress_len / factor, nkvh, dk});
            auto v_comp_head = v_comp2d->view({compress_len / factor, nkvh, dk});

            if (remainder_len == 0) {
                return {k_comp_head, v_comp_head};
            }

            std::shared_ptr<Tensor> k_comp;
            std::shared_ptr<Tensor> v_comp;
            k_comp = Tensor::buffer(k_tensor->dtype(),
                              {compressed_seq_len + remainder_len, nkvh, dk},
                              ctx_ptr->memory_pool);

            v_comp = Tensor::buffer(v_tensor->dtype(),
                              {compressed_seq_len + remainder_len, nkvh, dk},
                              ctx_ptr->memory_pool);

            auto k_tail = k_seg->slice(0, compress_len, remainder_len);
            auto v_tail = v_seg->slice(0, compress_len, remainder_len);
            // 目标前半段 [0, compressed_seq_len) 放压缩后的 head
            auto k_dst_head = k_comp->slice(0, 0, compressed_seq_len);
            auto v_dst_head = v_comp->slice(0, 0, compressed_seq_len);
            rearrange(k_dst_head, k_comp_head);  // [compressed_seq_len, nkvh, dk]
            rearrange(v_dst_head, v_comp_head);

            // 目标后半段 [compressed_seq_len, compressed_seq_len + remainder_len) 放 tail
            auto k_dst_tail = k_comp->slice(0, compressed_seq_len, remainder_len);
            auto v_dst_tail = v_comp->slice(0, compressed_seq_len, remainder_len);
            rearrange(k_dst_tail, k_tail);       // [remainder_len, nkvh, dk]
            rearrange(v_dst_tail, v_tail);
            // auto k_out = Tensor::buffer(k_seg->dtype(), {compress_len / factor + remainder_len, nkvh, dk}, ctx_ptr->memory_pool);
            // auto v_out = Tensor::buffer(v_seg->dtype(), {compress_len / factor + remainder_len, nkvh, dk}, ctx_ptr->memory_pool);

            // RUN_INFINI(infinirtMemcpy(k_out->data(), k_comp_head->data(),
            //                           k_comp_head->numel() * dsize(k_comp_head->dtype()),
            //                           INFINIRT_MEMCPY_D2D));
            // RUN_INFINI(infinirtMemcpy(v_out->data(), v_comp_head->data(),
            //                           v_comp_head->numel() * dsize(v_comp_head->dtype()),
            //                           INFINIRT_MEMCPY_D2D));
            // auto head_elems = k_comp_head->numel();
            // RUN_INFINI(infinirtMemcpy(k_out->data(head_elems * dsize(k_out->dtype())),
            //                           k_tail->data(),
            //                           k_tail->numel() * dsize(k_tail->dtype()),
            //                           INFINIRT_MEMCPY_D2D));
            // RUN_INFINI(infinirtMemcpy(v_out->data(head_elems * dsize(v_out->dtype())),
            //                           v_tail->data(),
            //                           v_tail->numel() * dsize(v_tail->dtype()),
            //                           INFINIRT_MEMCPY_D2D));
            return {k_comp, v_comp};
        };

        uint32_t img_len = std::min<uint32_t>(config_.image_kv_len, seq);
        uint32_t txt_len = seq - img_len;


        //这里可能有坑
        auto k_img = img_len > 0 ? k_tensor->slice(0, 0, img_len) : nullptr;
        auto v_img = img_len > 0 ? v_tensor->slice(0, 0, img_len) : nullptr;
        auto k_txt = txt_len > 0 ? k_tensor->slice(0, img_len, txt_len) : nullptr;
        auto v_txt = txt_len > 0 ? v_tensor->slice(0, img_len, txt_len) : nullptr;

        std::shared_ptr<Tensor> k_img_comp, v_img_comp, k_txt_comp, v_txt_comp;
        if (img_len > 0) {
            auto res = compress_segment(k_img, v_img, 2); // compress_ik/iv
            k_img_comp = res.first;
            v_img_comp = res.second;
        }
        if (txt_len > 0) {
            auto res = compress_segment(k_txt, v_txt, 0); // compress_tk/tv
            k_txt_comp = res.first;
            v_txt_comp = res.second;
        }

        std::shared_ptr<Tensor> k_comp, v_comp;
        if (k_img_comp && k_txt_comp) {
            auto total_len = k_img_comp->shape()[0] + k_txt_comp->shape()[0];
            k_comp = Tensor::buffer(k_tensor->dtype(), {total_len, nkvh, dk}, ctx_ptr->memory_pool);
            v_comp = Tensor::buffer(v_tensor->dtype(), {total_len, nkvh, dk}, ctx_ptr->memory_pool);
            // concat along seq dim using slice+rearrange
            auto k_dst_img = k_comp->slice(0, 0, k_img_comp->shape()[0]);
            auto k_dst_txt = k_comp->slice(0, k_img_comp->shape()[0], k_txt_comp->shape()[0]);
            auto v_dst_img = v_comp->slice(0, 0, v_img_comp->shape()[0]);
            auto v_dst_txt = v_comp->slice(0, v_img_comp->shape()[0], v_txt_comp->shape()[0]);
            rearrange(k_dst_img, k_img_comp);
            rearrange(k_dst_txt, k_txt_comp);
            rearrange(v_dst_img, v_img_comp);
            rearrange(v_dst_txt, v_txt_comp);
        } else {
            k_comp = k_img_comp ? k_img_comp : k_txt_comp;
            v_comp = v_img_comp ? v_img_comp : v_txt_comp;
        }

        compressed->layers[layer].k_comp = k_comp;
        compressed->layers[layer].v_comp = v_comp;
        compressed->layers[layer].orig_seq_len = seq;
        compressed->layers[layer].comp_seq_len = k_comp ? static_cast<uint32_t>(k_comp->shape()[0]) : 0;
    }

    return compressed;
}





bool Compressor::decompress(const CompressedKV &ckv,
                            std::vector<std::shared_ptr<Tensor>> &k_out,
                            std::vector<std::shared_ptr<Tensor>> &v_out) {
    // Placeholder: no real decompression; just fail to signal unimplemented.
    (void)ckv;
    (void)k_out;
    (void)v_out;
    return false;
}

// Optional helper: create a placeholder compressor that is disabled.
std::unique_ptr<Compressor> createDisabledCompressor() {
    CompressionConfig cfg;
    cfg.enable = false;
    return std::make_unique<Compressor>(cfg);
}
