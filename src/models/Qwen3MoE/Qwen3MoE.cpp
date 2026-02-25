#include "Qwen3MoE_impl.hpp"
#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer.h"
#include <random>
#include <thread>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
// =============================================================================
// Helper Declarations & Utils
// =============================================================================

void createDeviceResource(Qwen3MoEDeviceResource *rsrc, 
    const Qwen3MoEAttentionMeta *meta,
    std::shared_ptr<Qwen3DeviceWeights> weights, 
    infiniDevice_t device, int idev,
    int ndev, int dev_id,
    infinicclComm_t comm) {

    RUN_INFINI(infinirtSetDevice(device, dev_id));
    RUN_INFINI(infinirtStreamSynchronize(weights->load_stream));

    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);

    infinirtStream_t stream;
    infinirtStreamCreate(&stream); 

    auto memory_pool = std::make_shared<MemoryPool>();

    *rsrc = Qwen3MoEDeviceResource{
    device,
    dev_id,
    handle,
    weights,
    stream,
    comm,
    memory_pool,
    };

    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(Qwen3MoEDeviceResource &res) {
    infinirtDeviceSynchronize();
    res.weights.reset();
    if (res.handle) { infiniopDestroyHandle(res.handle); res.handle = nullptr; }
    if (res.stream) { infinirtStreamDestroy(res.stream); res.stream = nullptr; }
    if (res.comm) { infinicclCommDestroy(res.comm); res.comm = nullptr; }
}

// =============================================================================
// Inference Logic
// =============================================================================

// Qwen3MoE.cpp

void inferBatchQwen3MoE(const Qwen3MoEAttentionMeta &meta,
    Qwen3MoEDeviceResource &rsrc,
    std::shared_ptr<Tensor> input_hidden_states,
    std::shared_ptr<Tensor> pos_ids,
    std::shared_ptr<Tensor> output_tensor,
    Qwen3Cache *kv_cache,
    size_t layer_id,
    int batch_size,
    const std::vector<int>& _seq_lens, 
    const std::vector<int>& _past_lens
) {
    infiniopHandle_t handle = rsrc.handle;
    infinirtStream_t stream = rsrc.stream;
    auto memory_pool = rsrc.memory_pool;
    auto dt_logits = meta.dtype;

    const auto &layer_weight = rsrc.weights->w_layers[layer_id];
    const auto &attn_weight = layer_weight.self_attn;

    // [FINAL TRUTH] Based on weight shape [4096, 2048]
    size_t num_heads = 32;     
    size_t num_kv_head = 4;    
    size_t head_dim = 128;     
    size_t ngroup = num_heads / num_kv_head; // 8

    auto input_shape = input_hidden_states->shape();
    size_t ntok = input_shape[0];

    std::vector<int> seq_lens = _seq_lens;
    std::vector<int> past_lens = _past_lens;
    std::vector<int> cpu_pos_ids(ntok);

    RUN_INFINI(infinirtMemcpyAsync(cpu_pos_ids.data(), pos_ids->data(), ntok * sizeof(int), INFINIRT_MEMCPY_D2H, stream));
    RUN_INFINI(infinirtStreamSynchronize(stream));

    size_t pos_offset = 0;
    for (int b = 0; b < batch_size; ++b) {
    int current_pos = cpu_pos_ids[pos_offset];
    if (past_lens[b] == 0 && current_pos > 0) {
        past_lens[b] = current_pos;
        }
        pos_offset += seq_lens[b];
        }

    CacheManager cache_manager(100);
    InferenceContext ctx(handle, memory_pool, &cache_manager, stream);
    setInferenceContext(&ctx);

    // Alloc Buffers (Full 128-dim size)
    // Q: 32 * 128 = 4096
    // K/V: 4 * 128 = 512
    auto q_buf = Tensor::buffer(dt_logits, {ntok, num_heads * head_dim}, memory_pool);
    auto k_buf = Tensor::buffer(dt_logits, {ntok, num_kv_head * head_dim}, memory_pool);
    auto v_buf = Tensor::buffer(dt_logits, {ntok, num_kv_head * head_dim}, memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, num_heads * head_dim}, memory_pool);

    // Step 1: Projections
    linear(q_buf, input_hidden_states, attn_weight->q_proj, 1.f, 0.f, nullptr, nullptr);
    linear(k_buf, input_hidden_states, attn_weight->k_proj, 1.f, 0.f, nullptr, nullptr);
    linear(v_buf, input_hidden_states, attn_weight->v_proj, 1.f, 0.f, nullptr, nullptr);

    int check_pos_id = 64;
    size_t half_dim = 64; // head_dim / 2
    std::vector<unsigned short> h_cos_row(half_dim);
    
    // Offset = row * row_stride (half_dim elements)
    size_t cos_offset = check_pos_id * half_dim;
    
    // Assuming cos_table is BF16
    RUN_INFINI(infinirtMemcpyAsync(h_cos_row.data(), 
                    (char*)rsrc.weights->cos_table->data() + cos_offset * sizeof(unsigned short), 
                    half_dim * sizeof(unsigned short), 
                    INFINIRT_MEMCPY_D2H, 
                    stream));
    RUN_INFINI(infinirtStreamSynchronize(stream));

    // Step 2: QK Norm (128-dim)
    {
        auto q_norm_view = q_buf->view({ntok, num_heads, head_dim});
        auto k_norm_view = k_buf->view({ntok, num_kv_head, head_dim});

    if (rsrc.weights->w_layers[layer_id].self_attn->q_norm) {
        rmsnorm(q_norm_view, q_norm_view, rsrc.weights->w_layers[layer_id].self_attn->q_norm, 1e-6);
    }
    if (rsrc.weights->w_layers[layer_id].self_attn->k_norm) {
        rmsnorm(k_norm_view, k_norm_view, rsrc.weights->w_layers[layer_id].self_attn->k_norm, 1e-6);
    }
    }
    
    // Step 3: RoPE (128-dim)
    {
        auto q_rope = q_buf->view({ntok, num_heads, head_dim});
        auto k_rope = k_buf->view({ntok, num_kv_head, head_dim});
        
        rope_v2(q_rope, q_rope, pos_ids, rsrc.weights->cos_table, rsrc.weights->sin_table);
        rope_v2(k_rope, k_rope, pos_ids, rsrc.weights->cos_table, rsrc.weights->sin_table);
    }

    // =========================================================
    // Step 4: KV Cache Setup & Batch Loop
    // =========================================================
        
        // 1. KV Cache Initialization
        if (kv_cache->layers.size() <= layer_id) {
            kv_cache->layers.resize(layer_id + 1);
        }
        auto &kv_cache_layer = kv_cache->layers[layer_id];
        size_t max_seq_len = meta.max_seq_len;
        
        // [RESTORED STANDARD LOGIC]
        // 只有当指针为空，或者形状不匹配时，才重新分配！
        bool need_alloc = false;
        if (!kv_cache_layer.first || !kv_cache_layer.second) {
            need_alloc = true;
        } else {
            auto s = kv_cache_layer.first->shape();
            if (s[0] < static_cast<size_t>(batch_size) || 
                s[1] != num_kv_head || 
                s[2] != max_seq_len || 
                s[3] != head_dim) { 
                need_alloc = true;
            }
        }
        size_t unit_size = dsize(dt_logits);
        if (need_alloc) {
            kv_cache_layer.first = Tensor::buffer(dt_logits, {static_cast<size_t>(batch_size), num_kv_head, max_seq_len, head_dim}, memory_pool);
            kv_cache_layer.second = Tensor::buffer(dt_logits, {static_cast<size_t>(batch_size), num_kv_head, max_seq_len, head_dim}, memory_pool);
            
            // [REVERTED] Use cudaMemsetAsync (Stable)
            size_t num_elements = static_cast<size_t>(batch_size) * num_kv_head * max_seq_len * head_dim;
            size_t total_bytes = num_elements * unit_size;
            
            // [SAFEGUARD] Check size > 0
            if (total_bytes > 0) {
                cudaMemsetAsync(kv_cache_layer.first->data(), 0, total_bytes, (cudaStream_t)stream);
                cudaMemsetAsync(kv_cache_layer.second->data(), 0, total_bytes, (cudaStream_t)stream);
            }
        } 

        auto k_cache_all = kv_cache_layer.first;
        auto v_cache_all = kv_cache_layer.second;


    char* k_cache_base = (char*)k_cache_all->data();
    char* v_cache_base = (char*)v_cache_all->data();

    size_t stride_seq_bytes   = head_dim * unit_size;
    size_t stride_head_bytes  = max_seq_len * stride_seq_bytes;
    size_t stride_batch_bytes = num_kv_head * stride_head_bytes;

    size_t token_offset = 0; 

    for (int b = 0; b < batch_size; ++b) {
    size_t cur_seq_len = static_cast<size_t>(seq_lens[b]);
    size_t cur_past_len = static_cast<size_t>(past_lens[b]);
    size_t total_len = cur_past_len + cur_seq_len;

    // --- Cache Update ---
    char* k_src_batch_ptr = (char*)k_buf->data() + token_offset * num_kv_head * head_dim * unit_size;
    char* v_src_batch_ptr = (char*)v_buf->data() + token_offset * num_kv_head * head_dim * unit_size;
    char* k_dst_batch_base = k_cache_base + b * stride_batch_bytes;
    char* v_dst_batch_base = v_cache_base + b * stride_batch_bytes;
    size_t kv_token_bytes = head_dim * unit_size; 
    size_t src_pitch = num_kv_head * head_dim * unit_size; 
    size_t dst_pitch = head_dim * unit_size; 

    for (size_t h = 0; h < num_kv_head; h++) {
    char* k_s = k_src_batch_ptr + h * kv_token_bytes;
    char* v_s = v_src_batch_ptr + h * kv_token_bytes;
    char* k_d = k_dst_batch_base + h * stride_head_bytes + cur_past_len * stride_seq_bytes;
    char* v_d = v_dst_batch_base + h * stride_head_bytes + cur_past_len * stride_seq_bytes;
    cudaMemcpy2DAsync(k_d, dst_pitch, k_s, src_pitch, kv_token_bytes, cur_seq_len, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    cudaMemcpy2DAsync(v_d, dst_pitch, v_s, src_pitch, kv_token_bytes, cur_seq_len, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    }

    // --- Attention Compute  ---

    // 1. Prepare Q
    auto q_transposed = Tensor::buffer(dt_logits, {num_heads, cur_seq_len, head_dim}, memory_pool);
    auto q_src_view = q_buf->view({ntok, num_heads, head_dim})->slice(0, token_offset, cur_seq_len);
    for (size_t h = 0; h < num_heads; h++) {
    auto q_s = q_src_view->slice(1, h, 1)->view({cur_seq_len, head_dim});
    auto q_d = q_transposed->slice(0, h, 1)->view({cur_seq_len, head_dim});
    rearrange(q_d, q_s);
    }
    auto q_gemm = q_transposed->view({num_kv_head, ngroup * cur_seq_len, head_dim});

    // 2. Prepare K
    size_t padded_len = (total_len + 31) / 32 * 32; 
    auto k_padded_gather = Tensor::buffer(dt_logits, {num_kv_head, padded_len, head_dim}, memory_pool);
    size_t kv_gather_bytes = num_kv_head * padded_len * head_dim * unit_size;

    // [REVERTED] Use cudaMemsetAsync
    if (kv_gather_bytes > 0) {
        cudaMemsetAsync(k_padded_gather->data(), 0, kv_gather_bytes, (cudaStream_t)stream);
    }

    char* k_gather_src_base = k_cache_base + b * stride_batch_bytes;
    size_t gather_bytes_per_head = total_len * head_dim * unit_size;
    size_t dst_head_stride_bytes = padded_len * head_dim * unit_size;
    for (size_t h = 0; h < num_kv_head; h++) {
    char* k_src = k_gather_src_base + h * stride_head_bytes;
    char* k_dst = (char*)k_padded_gather->data() + h * dst_head_stride_bytes;
    // Keep size check for memcpy
    if (gather_bytes_per_head > 0) {
        cudaMemcpyAsync(k_dst, (void*)k_src, gather_bytes_per_head, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    }
    }

    auto k_gemm_in = Tensor::buffer(dt_logits, {num_kv_head, head_dim, padded_len}, memory_pool);
    rearrange(k_gemm_in, k_padded_gather->permute({0, 2, 1}));

    // 3. GEMM 1: Q * K
    auto scores_padded = Tensor::buffer(dt_logits, {num_kv_head, ngroup * cur_seq_len, padded_len}, memory_pool);

    // [Scheme A] Zero out the buffer safely
    size_t scores_bytes = num_kv_head * ngroup * cur_seq_len * padded_len * unit_size;
    cudaMemsetAsync(scores_padded->data(), 0, scores_bytes, (cudaStream_t)stream);

    float scale_factor = 1.0f / sqrt(128.0f); 
    linear(scores_padded, q_gemm, k_gemm_in, scale_factor, 0.f, nullptr, nullptr);

    // 4. Softmax+Scaling+Masking
    auto scores_view = scores_padded->view({num_heads, cur_seq_len, padded_len});
    auto scores_in = scores_view->slice(2, 0, total_len);
    causalSoftmax(scores_in, scores_in);

    if (padded_len > total_len) {
        size_t pitch = padded_len * unit_size;
        size_t width = (padded_len - total_len) * unit_size;
        char* dst_ptr = (char*)scores_padded->data() + total_len * unit_size;
        // Keep size check for 2D Memset
        if (width > 0) {
            cudaMemset2DAsync(dst_ptr, pitch, 0, width, num_heads * cur_seq_len, (cudaStream_t)stream);
        }
    }
    

    // 5. GEMM 2
    auto v_padded_gather = Tensor::buffer(dt_logits, {num_kv_head, padded_len, head_dim}, memory_pool);
    // [REVERTED] Use cudaMemsetAsync
    if (kv_gather_bytes > 0) {
        cudaMemsetAsync(v_padded_gather->data(), 0, kv_gather_bytes, (cudaStream_t)stream);
    }

    char* v_gather_src_base = v_cache_base + b * stride_batch_bytes;
    for (size_t h = 0; h < num_kv_head; h++) {
    char* v_src = v_gather_src_base + h * stride_head_bytes;
    char* v_dst = (char*)v_padded_gather->data() + h * dst_head_stride_bytes;
    // Keep size check for memcpy
    if (gather_bytes_per_head > 0) {
        cudaMemcpyAsync(v_dst, (void*)v_src, gather_bytes_per_head, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    }
    }

    auto attn_out_b = Tensor::buffer(dt_logits, {num_kv_head, ngroup * cur_seq_len, head_dim}, memory_pool);
    linear(attn_out_b, scores_padded, v_padded_gather, 1.f, 0.f, nullptr, nullptr);

    // Rearrange
    auto attn_out_view_flat = attn_out_b->view({num_heads, cur_seq_len, head_dim});
    auto o_dst_flat = o_buf->view({ntok, num_heads, head_dim})->slice(0, token_offset, cur_seq_len);
    for (size_t h = 0; h < num_heads; h++) {
    auto src_h = attn_out_view_flat->slice(0, h, 1)->view({cur_seq_len, head_dim});
    auto dst_h = o_dst_flat->slice(1, h, 1)->view({cur_seq_len, head_dim});
    rearrange(dst_h, src_h);
    }

    token_offset += cur_seq_len;
    } // End of Batch Loop

    // Step 6: Final Output Projection
    if (output_tensor) {
    size_t context_dim = num_heads * head_dim; 
    auto ctx_flat = o_buf->view({ntok, context_dim});
    auto w_o = attn_weight->o_proj; 
    size_t hidden_dim = meta.hidden_size; 
    auto out_flat = output_tensor->view({ntok, hidden_dim});
    linear(out_flat, ctx_flat, w_o, 1.0f, 0.0f, nullptr, nullptr);
    }
}

// =============================================================================
// Interface Exports
// =============================================================================

Qwen3MoEAttention::Qwen3MoEAttention(const Qwen3MoEAttentionMeta *_meta, const Qwen3MoEWeights *weights) : meta(*_meta) {
    auto device_weights = weights->device_weights;
    int ndev = device_weights.size();
    device = device_weights[0]->device;
    dev_ids.resize(ndev);
    for (int i = 0; i < ndev; i++) {
        dev_ids[i] = device_weights[i]->dev_id;
    }
    dev_resources = std::vector<Qwen3MoEDeviceResource>(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }
    for (int i = 0; i < ndev; i++) {
        createDeviceResource(&dev_resources[i], &meta, device_weights[i], device, i, ndev, dev_ids[i], comms[i]);
    }
}

__C __export struct Qwen3MoEAttention *createQwen3MoEAttention(const Qwen3MoEAttentionMeta *_meta,
                        const Qwen3MoEWeights *weights) {
    Qwen3MoEAttention *attention = new Qwen3MoEAttention(_meta, weights);
    return attention;
}

__C __export void destroyQwen3MoEAttention(struct Qwen3MoEAttention *ctx) {
    if (!ctx) return;
    auto ndev = ctx->dev_resources.size();
    for (size_t idev = 0; idev < ndev; idev++) {
        releaseDeviceResource(ctx->dev_resources[idev]);
    }
    delete ctx;
}

__C __export void forwardQwen3MoEAttention(
    struct Qwen3MoEAttention* context,
    struct Qwen3Cache* kv_cache,
    const void* input_tensor,
    void* output_tensor,
    int batch_size,
    const int* seq_lens_ptr,
    const int* past_lens_ptr,
    const int* pos_ids_ptr
) {
    if (!context || !kv_cache || !input_tensor || !output_tensor) {
        return;
    }
    
    size_t layer_id = 0;
    if (context->dev_resources.empty()) return;
    auto &rsrc = context->dev_resources[0];
    auto meta = &context->meta;
    auto dt_logits = meta->dtype;
    size_t hidden_size = meta->hidden_size;
    
    std::vector<int> seq_lens(batch_size);
    std::vector<int> past_lens(batch_size);
    std::memcpy(seq_lens.data(), seq_lens_ptr, batch_size * sizeof(int));
    std::memcpy(past_lens.data(), past_lens_ptr, batch_size * sizeof(int));
    
    size_t ntok = 0;
    for (int len : seq_lens) ntok += len;
    
    std::shared_ptr<Tensor> input_hidden_states;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        input_hidden_states = Tensor::weight(const_cast<void*>(input_tensor), dt_logits, {ntok, hidden_size});
    } else {
        input_hidden_states = Tensor::buffer(dt_logits, {ntok, hidden_size}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(input_hidden_states->data(), const_cast<void*>(input_tensor),
                                     dsize(dt_logits) * ntok * hidden_size,
                                     INFINIRT_MEMCPY_H2D, rsrc.stream));
    }
    
    std::shared_ptr<Tensor> pos_ids;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids = Tensor::weight(const_cast<int*>(pos_ids_ptr), INFINI_DTYPE_I32, {ntok});
    } else {
        pos_ids = Tensor::buffer(INFINI_DTYPE_I32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids->data(), (void*)pos_ids_ptr,
                                     sizeof(int) * ntok,
                                     INFINIRT_MEMCPY_H2D, rsrc.stream));
    }
    
    auto output_tensor_ptr = Tensor::buffer(dt_logits, {ntok, hidden_size}, rsrc.memory_pool);
    Qwen3Cache *qwen3_cache = reinterpret_cast<Qwen3Cache*>(kv_cache);
    inferBatchQwen3MoE(context->meta, rsrc, input_hidden_states, pos_ids,
                       output_tensor_ptr, qwen3_cache, layer_id,
                       batch_size, seq_lens, past_lens);

    RUN_INFINI(infinirtStreamSynchronize(rsrc.stream));

    if (rsrc.device != INFINI_DEVICE_CPU) {
        RUN_INFINI(infinirtMemcpyAsync(output_tensor, output_tensor_ptr->data(),
        dsize(dt_logits) * ntok * hidden_size,
        INFINIRT_MEMCPY_D2H, rsrc.stream));
    }
}

__C __export void injectQwen3CacheKV(
    struct Qwen3MoEAttention* context,
    struct Qwen3Cache* kv_cache,
    int layer_id,
    int batch_idx,
    int past_len,
    const void* k_host_ptr, 
    const void* v_host_ptr
) {
    if (!context || !kv_cache || past_len <= 0) return;

    auto &rsrc = context->dev_resources[0];
    RUN_INFINI(infinirtSetDevice(rsrc.device, rsrc.device_id));
    auto meta = &context->meta;
    auto memory_pool = rsrc.memory_pool;
    auto stream = rsrc.stream;
    
    if (kv_cache->layers.size() <= static_cast<size_t>(layer_id)) {
        kv_cache->layers.resize(static_cast<size_t>(layer_id) + 1);
    }
    auto &layer = kv_cache->layers[layer_id];
    
    size_t required_batch = batch_idx + 1;
    size_t H = meta->num_kv_head;
    size_t S = meta->max_seq_len;
    size_t D = meta->head_dim;
    
    bool need_alloc = false;
    if (!layer.first || !layer.second) {
        need_alloc = true;
    } else {
        if (layer.first->shape()[0] < required_batch) need_alloc = true;
    }
    
    // [FIX] Force minimum allocation size to avoid mid-loop resizing/resetting
    size_t current_capacity = 0;
    if (layer.first) current_capacity = layer.first->shape()[0];
    size_t target_capacity = std::max(required_batch, (size_t)16);
    
    if (current_capacity < target_capacity) {
        need_alloc = true;
    }
    
    if (need_alloc) {
        layer.first = Tensor::buffer(meta->dtype, {target_capacity, H, S, D}, memory_pool);
        layer.second = Tensor::buffer(meta->dtype, {target_capacity, H, S, D}, memory_pool);
        RUN_INFINI(infinirtStreamSynchronize(stream));
    }

    auto k_tensor = layer.first;
    auto v_tensor = layer.second;

    
    size_t dtype_size = dsize(meta->dtype);
    size_t stride_seq_bytes   = D * dtype_size;
    size_t stride_head_bytes  = S * stride_seq_bytes;
    size_t stride_batch_bytes = H * stride_head_bytes;

    char* k_base = (char*)k_tensor->data();
    char* v_base = (char*)v_tensor->data();
    
    char* k_batch_base = k_base + batch_idx * stride_batch_bytes;
    char* v_batch_base = v_base + batch_idx * stride_batch_bytes;

    const char* k_src_base = (const char*)k_host_ptr;
    const char* v_src_base = (const char*)v_host_ptr;
    
    size_t src_head_stride_bytes = past_len * D * dtype_size;
    size_t bytes_to_copy_per_head = past_len * D * dtype_size;

    for (size_t h = 0; h < H; ++h) {
        char* k_dst_addr = k_batch_base + h * stride_head_bytes;
        char* v_dst_addr = v_batch_base + h * stride_head_bytes;

        const char* k_src_addr = k_src_base + h * src_head_stride_bytes;
        const char* v_src_addr = v_src_base + h * src_head_stride_bytes;

        if (rsrc.device == INFINI_DEVICE_CPU) {
            std::memcpy(k_dst_addr, k_src_addr, bytes_to_copy_per_head);
            std::memcpy(v_dst_addr, v_src_addr, bytes_to_copy_per_head);
        } else {
            if (bytes_to_copy_per_head > 0) {
                RUN_INFINI(infinirtMemcpyAsync(k_dst_addr, (void*)k_src_addr,
                            bytes_to_copy_per_head, INFINIRT_MEMCPY_H2D, stream));
                RUN_INFINI(infinirtMemcpyAsync(v_dst_addr, (void*)v_src_addr,
                            bytes_to_copy_per_head, INFINIRT_MEMCPY_H2D, stream));
            }
        }
    }
    RUN_INFINI(infinirtStreamSynchronize(stream));
}

extern "C" void customInjectCacheKV(
    Qwen3Cache *kv_cache,
    size_t layer_id,
    int batch_idx,
    int past_len,
    void* k_src_ptr,
    void* v_src_ptr,
    cudaStream_t stream
) {
    int dev_id = 0;
    cudaGetDevice(&dev_id);
    RUN_INFINI(infinirtSetDevice(INFINI_DEVICE_NVIDIA, dev_id));

    // 1. 安全检查
    if (!kv_cache || kv_cache->layers.size() <= layer_id) {
        std::cout<< "检查 unpass!" << std::endl;
        return;

    }
    
    auto &layer = kv_cache->layers[layer_id];
    //std::cout<< layer_id << std::endl;
    // 如果显存还没分配（Dummy Forward 没跑？），直接返回，Python侧会报错
    if (!layer.first || !layer.second) {
        printf(">>> [C++ Error] Cache not allocated yet! Run dummy forward first.\n");
        return;
    }

    // 2. 获取 C++ 视角的形状信息
    auto shape = layer.first->shape(); 
    // shape: [Batch, NumKV, MaxSeq, HeadDim]
    size_t num_kv = shape[1];
    size_t max_seq = shape[2]; // 这里是关键！它是 8192
    size_t head_dim = shape[3]; // 这里应该是 128

    // 3. 计算 C++ 显存中的 Stride (稀疏布局)
    size_t dtype_size = 2; // BF16 = 2 bytes
    size_t stride_seq = head_dim * dtype_size;
    size_t stride_head = max_seq * stride_seq;       // 跨越 8192 个 Token
    size_t stride_batch = num_kv * stride_head;

    // 4. 计算目标地址基址 (Base Address for this specific Batch)
    char* k_dst_base = (char*)layer.first->data() + batch_idx * stride_batch;
    char* v_dst_base = (char*)layer.second->data() + batch_idx * stride_batch;

    // 5. 搬运循环
    // Python 传来的数据是紧凑的: [NumKV, PastLen, HeadDim]
    // 我们需要把每个 Head 的 [PastLen, HeadDim] 块搬运过去
    
    size_t copy_bytes_per_head = past_len * head_dim * dtype_size;
    size_t src_stride_head = copy_bytes_per_head; // Python端是紧凑的

    for (size_t h = 0; h < num_kv; ++h) {
        // Source: Python (Compact)
        char* k_src = (char*)k_src_ptr + h * src_stride_head;
        char* v_src = (char*)v_src_ptr + h * src_stride_head;

        // Dest: C++ (Sparse / Strided)
        // 注意：我们从 sequence 的 index 0 开始写起
        //int start_pos = past_len;
        char* k_dst = k_dst_base + h * stride_head ;
        char* v_dst = v_dst_base + h * stride_head ;

        // 检查指针是否对齐和越界（简单保护）
        if (past_len > 0) {
             RUN_INFINI(infinirtMemcpyAsync(k_dst, k_src, copy_bytes_per_head, INFINIRT_MEMCPY_H2D, (infinirtStream_t)stream));
             RUN_INFINI(infinirtMemcpyAsync(v_dst, v_src, copy_bytes_per_head, INFINIRT_MEMCPY_H2D, (infinirtStream_t)stream));
        }
    }
    
    // 简单同步确保写入完成
    RUN_INFINI(infinirtStreamSynchronize((infinirtStream_t)stream));
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("DEBUG: Error at customInjectCacheKV end: %s\n", cudaGetErrorString(err));
    }
}