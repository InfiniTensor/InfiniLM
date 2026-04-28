#ifndef __PAGED_ATTENTION_PREFILL_KERNEL_V2_CUH__
#define __PAGED_ATTENTION_PREFILL_KERNEL_V2_CUH__

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ALI_API) || defined(ENABLE_ILUVATAR_API)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#endif

#include <cstdint>
#include <type_traits>

// Reuse warp-level primitives and math helpers from decode flash_attention kernels.
#include "../../paged_attention/cuda/kernel_v2.cuh"

namespace op::paged_attention_prefill::cuda {

template <typename Tindex>
__device__ __forceinline__ size_t find_seq_id(size_t token_idx, const Tindex *cu_seqlens_q, size_t num_seqs) {
    size_t low = 0, high = (num_seqs == 0) ? 0 : (num_seqs - 1);
    while (low <= high) {
        size_t mid = (low + high) >> 1;
        const size_t start = static_cast<size_t>(cu_seqlens_q[mid]);
        const size_t end = static_cast<size_t>(cu_seqlens_q[mid + 1]);
        if (token_idx >= start && token_idx < end) {
            return mid;
        } else if (token_idx < start) {
            if (mid == 0) {
                break;
            }
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return 0;
}

template <typename Tindex, typename Tdata, int HEAD_SIZE>
__device__ void PagedAttentionPrefillWarpKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cu_seqlens_q_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {

    constexpr int kWarpSize = 32;
    static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128, "Only head_size 64/128 supported in v0.4.");
    static_assert(HEAD_SIZE % kWarpSize == 0, "HEAD_SIZE must be divisible by 32.");
    constexpr int DIMS_PER_THREAD = HEAD_SIZE / kWarpSize;

    const int lane = threadIdx.x;

    const int head_idx = static_cast<int>(blockIdx.x);
    const int seq_idx = static_cast<int>(blockIdx.y);
    const int q_token_local = static_cast<int>(blockIdx.z);

    const Tindex q_start = cu_seqlens_q_[seq_idx];
    const Tindex q_end = cu_seqlens_q_[seq_idx + 1];
    const int q_len = static_cast<int>(q_end - q_start);
    if (q_token_local >= q_len) {
        return;
    }

    const int kv_len_total = static_cast<int>(total_kv_lens_[seq_idx]);
    const int history_len = kv_len_total - q_len;
    const int allowed_k_len = history_len + q_token_local + 1;
    if (allowed_k_len <= 0) {
        return;
    }

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    const int64_t q_token = q_start + static_cast<int64_t>(q_token_local);
    const Tdata *q_ptr = q_ + q_token * q_stride + static_cast<int64_t>(head_idx) * q_head_stride;
    Tdata *out_ptr = out_ + q_token * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;

    const Tindex *block_table = block_tables_ + static_cast<int64_t>(seq_idx) * static_cast<int64_t>(block_table_batch_stride);

    float q_reg[DIMS_PER_THREAD];
    float acc[DIMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        q_reg[i] = static_cast<float>(q_ptr[dim]);
        acc[i] = 0.0f;
    }

#if defined(__CUDA_ARCH__)
    float2 q_reg2[DIMS_PER_THREAD / 2];
    if constexpr (std::is_same_v<Tdata, half>) {
        const int dim_base = lane * DIMS_PER_THREAD;
        const half2 *q2 = reinterpret_cast<const half2 *>(q_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
            q_reg2[j] = __half22float2(q2[j]);
        }
    }
    if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
        const int dim_base = lane * DIMS_PER_THREAD;
        const __nv_bfloat162 *q2 = reinterpret_cast<const __nv_bfloat162 *>(q_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
            q_reg2[j] = __bfloat1622float2(q2[j]);
        }
    }
#endif

    float m = -INFINITY;
    float l = 0.0f;

    const int pbs = static_cast<int>(page_block_size);
    int t_base = 0;
    for (int logical_block = 0; t_base < allowed_k_len; ++logical_block, t_base += pbs) {
        int physical_block = 0;
        if (lane == 0) {
            physical_block = static_cast<int>(block_table[logical_block]);
        }
        physical_block = __shfl_sync(0xffffffff, physical_block, 0);

        const Tdata *k_base = k_cache_ + static_cast<int64_t>(physical_block) * k_batch_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;
        const Tdata *v_base = v_cache_ + static_cast<int64_t>(physical_block) * v_batch_stride + static_cast<int64_t>(kv_head_idx) * v_head_stride;

        const int token_end = min(pbs, allowed_k_len - t_base);
        for (int token_in_block = 0; token_in_block < token_end; ++token_in_block) {
            const int t = t_base + token_in_block;
            const Tdata *k_ptr = k_base + static_cast<int64_t>(token_in_block) * k_row_stride;
            const Tdata *v_ptr = v_base + static_cast<int64_t>(token_in_block) * v_row_stride;

            float qk = 0.0f;
#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<Tdata, half>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const half2 *k2 = reinterpret_cast<const half2 *>(k_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 qf = q_reg2[j];
                    const float2 kf = __half22float2(k2[j]);
                    qk += qf.x * kf.x + qf.y * kf.y;
                }
            } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const __nv_bfloat162 *k2 = reinterpret_cast<const __nv_bfloat162 *>(k_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 qf = q_reg2[j];
                    const float2 kf = __bfloat1622float2(k2[j]);
                    qk += qf.x * kf.x + qf.y * kf.y;
                }
            } else
#endif
#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    const int dim = lane * DIMS_PER_THREAD + i;
                    qk += q_reg[i] * static_cast<float>(k_ptr[dim]);
                }
            qk = op::paged_attention::cuda::warpReduceSum(qk);

            float alpha = 1.0f;
            float beta = 0.0f;
            if (lane == 0) {
                float score = qk * scale_log2;
                if (alibi_slope != 0.0f) {
                    const int causal_limit = allowed_k_len - 1;
                    score += (alibi_slope * static_cast<float>(t - causal_limit)) * kLog2e;
                }
                const float m_new = fmaxf(m, score);
                alpha = exp2f(m - m_new);
                beta = exp2f(score - m_new);
                l = l * alpha + beta;
                m = m_new;
            }
            alpha = op::paged_attention::cuda::warpBroadcast(alpha, 0);
            beta = op::paged_attention::cuda::warpBroadcast(beta, 0);

#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<Tdata, half>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const half2 *v2 = reinterpret_cast<const half2 *>(v_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 vf = __half22float2(v2[j]);
                    acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
                    acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
                }
            } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const __nv_bfloat162 *v2 = reinterpret_cast<const __nv_bfloat162 *>(v_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 vf = __bfloat1622float2(v2[j]);
                    acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
                    acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
                }
            } else
#endif
            {
#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    const int dim = lane * DIMS_PER_THREAD + i;
                    const float v_val = static_cast<float>(v_ptr[dim]);
                    acc[i] = acc[i] * alpha + beta * v_val;
                }
            }
        }
    }

    float inv_l = 0.0f;
    if (lane == 0) {
        inv_l = 1.0f / (l + 1e-6f);
    }
    inv_l = op::paged_attention::cuda::warpBroadcast(inv_l, 0);

#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        const float o = acc[i] * inv_l;
        if constexpr (std::is_same_v<Tdata, half>) {
            out_ptr[dim] = __float2half_rn(o);
        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
            out_ptr[dim] = __float2bfloat16_rn(o);
        } else {
            out_ptr[dim] = static_cast<Tdata>(o);
        }
    }
}

template <typename Tindex, typename Tdata, int HEAD_SIZE>
__global__ void PagedAttentionPrefillWarpGlobalKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cu_seqlens_q_,
    const float *alibi_slopes_,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t total_q_tokens,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {

    constexpr int kWarpSize = 32;
    static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128, "Only head_size 64/128 supported in v0.4.");
    static_assert(HEAD_SIZE % kWarpSize == 0, "HEAD_SIZE must be divisible by 32.");
    constexpr int DIMS_PER_THREAD = HEAD_SIZE / kWarpSize;

    const int lane = threadIdx.x;
    const size_t head_idx = static_cast<size_t>(blockIdx.x);
    const size_t global_token_idx = static_cast<size_t>(blockIdx.y);

    if (lane >= kWarpSize || head_idx >= num_heads || global_token_idx >= total_q_tokens) {
        return;
    }

    const size_t seq_idx = find_seq_id<Tindex>(global_token_idx, cu_seqlens_q_, num_seqs);
    const Tindex q_start = cu_seqlens_q_[seq_idx];
    const Tindex q_end = cu_seqlens_q_[seq_idx + 1];
    const int q_len = static_cast<int>(q_end - q_start);

    const int q_token_local = static_cast<int>(global_token_idx - static_cast<size_t>(q_start));
    if (q_token_local < 0 || q_token_local >= q_len) {
        return;
    }

    const int kv_len_total = static_cast<int>(total_kv_lens_[seq_idx]);
    const int history_len = kv_len_total - q_len;
    const int allowed_k_len = history_len + q_token_local + 1;
    if (allowed_k_len <= 0) {
        return;
    }

    const int num_queries_per_kv = static_cast<int>(num_heads / num_kv_heads);
    const int kv_head_idx = static_cast<int>(head_idx) / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    const Tdata *q_ptr = q_ + static_cast<int64_t>(global_token_idx) * q_stride + static_cast<int64_t>(head_idx) * q_head_stride;
    Tdata *out_ptr = out_ + static_cast<int64_t>(global_token_idx) * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;

    const Tindex *block_table = block_tables_ + static_cast<int64_t>(seq_idx) * static_cast<int64_t>(block_table_batch_stride);
    const int pbs = static_cast<int>(page_block_size);

    float q_reg[DIMS_PER_THREAD];
    float acc[DIMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        q_reg[i] = static_cast<float>(q_ptr[dim]);
        acc[i] = 0.0f;
    }

#if defined(__CUDA_ARCH__)
    float2 q_reg2[DIMS_PER_THREAD / 2];
    if constexpr (std::is_same_v<Tdata, half>) {
        const int dim_base = lane * DIMS_PER_THREAD;
        const half2 *q2 = reinterpret_cast<const half2 *>(q_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
            q_reg2[j] = __half22float2(q2[j]);
        }
    }
    if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
        const int dim_base = lane * DIMS_PER_THREAD;
        const __nv_bfloat162 *q2 = reinterpret_cast<const __nv_bfloat162 *>(q_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
            q_reg2[j] = __bfloat1622float2(q2[j]);
        }
    }
#endif

    float m = -INFINITY;
    float l = 0.0f;

    // Iterate by pages to avoid per-token division/mod and redundant block_table loads.
    int t_base = 0;
    for (int logical_block = 0; t_base < allowed_k_len; ++logical_block, t_base += pbs) {
        const int32_t phys = static_cast<int32_t>(block_table[logical_block]);
        const Tdata *k_base = k_cache_ + static_cast<int64_t>(phys) * k_batch_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;
        const Tdata *v_base = v_cache_ + static_cast<int64_t>(phys) * v_batch_stride + static_cast<int64_t>(kv_head_idx) * v_head_stride;

        const int token_end = min(pbs, allowed_k_len - t_base);
        for (int token_in_block = 0; token_in_block < token_end; ++token_in_block) {
            const int t = t_base + token_in_block;
            const Tdata *k_ptr = k_base + static_cast<int64_t>(token_in_block) * k_row_stride;
            const Tdata *v_ptr = v_base + static_cast<int64_t>(token_in_block) * v_row_stride;

            float qk = 0.0f;
#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<Tdata, half>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const half2 *k2 = reinterpret_cast<const half2 *>(k_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 qf = q_reg2[j];
                    const float2 kf = __half22float2(k2[j]);
                    qk += qf.x * kf.x + qf.y * kf.y;
                }
            } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const __nv_bfloat162 *k2 = reinterpret_cast<const __nv_bfloat162 *>(k_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 qf = q_reg2[j];
                    const float2 kf = __bfloat1622float2(k2[j]);
                    qk += qf.x * kf.x + qf.y * kf.y;
                }
            } else
#endif
            {
#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    const int dim = lane * DIMS_PER_THREAD + i;
                    qk += q_reg[i] * static_cast<float>(k_ptr[dim]);
                }
            }
            qk = op::paged_attention::cuda::warpReduceSum(qk);

            float alpha = 1.0f;
            float beta = 0.0f;
            if (lane == 0) {
                float score = qk * scale_log2;
                if (alibi_slope != 0.0f) {
                    const int causal_limit = allowed_k_len - 1;
                    score += (alibi_slope * static_cast<float>(t - causal_limit)) * kLog2e;
                }
                const float m_new = fmaxf(m, score);
                alpha = exp2f(m - m_new);
                beta = exp2f(score - m_new);
                l = l * alpha + beta;
                m = m_new;
            }
            alpha = op::paged_attention::cuda::warpBroadcast(alpha, 0);
            beta = op::paged_attention::cuda::warpBroadcast(beta, 0);

#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<Tdata, half>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const half2 *v2 = reinterpret_cast<const half2 *>(v_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 vf = __half22float2(v2[j]);
                    acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
                    acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
                }
            } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const __nv_bfloat162 *v2 = reinterpret_cast<const __nv_bfloat162 *>(v_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 vf = __bfloat1622float2(v2[j]);
                    acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
                    acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
                }
            } else
#endif
            {
#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    const int dim = lane * DIMS_PER_THREAD + i;
                    const float v_val = static_cast<float>(v_ptr[dim]);
                    acc[i] = acc[i] * alpha + beta * v_val;
                }
            }
        }
    }

    float inv_l = 0.0f;
    if (lane == 0) {
        inv_l = 1.0f / (l + 1e-6f);
    }
#ifdef ENABLE_ILUVATAR_API
    inv_l = op::paged_attention::cuda::warpBroadcast(inv_l, 0);
#else
    inv_l = __shfl_sync(0xffffffff, inv_l, 0);
#endif

#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        const float o = acc[i] * inv_l;
        if constexpr (std::is_same_v<Tdata, half>) {
            out_ptr[dim] = __float2half_rn(o);
        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
            out_ptr[dim] = __float2bfloat16_rn(o);
        } else {
            out_ptr[dim] = static_cast<Tdata>(o);
        }
    }
}

template <typename Tindex, typename Tdata, typename Tcompute, int HEAD_SIZE>
__global__ void PagedAttentionPrefillReferenceKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cu_seqlens_q_,
    const float *alibi_slopes_,
    size_t num_heads,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    size_t num_seqs) {

    const size_t global_token_idx = static_cast<size_t>(blockIdx.x);
    const size_t head_idx = static_cast<size_t>(blockIdx.y);
    const size_t dim_idx = static_cast<size_t>(threadIdx.x);

    if (dim_idx >= HEAD_SIZE || head_idx >= num_heads) {
        return;
    }

    const size_t seq_idx = find_seq_id<Tindex>(global_token_idx, cu_seqlens_q_, num_seqs);
    const size_t q_token_idx = global_token_idx - static_cast<size_t>(cu_seqlens_q_[seq_idx]);
    const size_t q_len = static_cast<size_t>(cu_seqlens_q_[seq_idx + 1] - cu_seqlens_q_[seq_idx]);

    const size_t total_kv_len = static_cast<size_t>(total_kv_lens_[seq_idx]);
    const size_t history_len = total_kv_len - q_len;
    const size_t causal_limit = history_len + q_token_idx;

    const size_t num_queries_per_kv = num_heads / num_kv_heads;
    const size_t kv_head_idx = head_idx / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];

    const Tdata *q_vec = q_ + static_cast<int64_t>(global_token_idx) * q_stride + static_cast<int64_t>(head_idx) * q_head_stride;
    Tdata *out_ptr = out_ + static_cast<int64_t>(global_token_idx) * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;

    const Tindex *block_table = block_tables_ + static_cast<int64_t>(seq_idx) * static_cast<int64_t>(block_table_batch_stride);
    const size_t pbs = page_block_size;

    Tcompute max_score = -INFINITY;
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t page = t / pbs;
        const size_t off = t - page * pbs;
        const ptrdiff_t phys = static_cast<ptrdiff_t>(block_table[page]);
        const Tdata *k_vec = k_cache_ + static_cast<int64_t>(phys) * k_batch_stride + static_cast<int64_t>(off) * k_row_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;

        Tcompute score = 0;
        for (size_t d = 0; d < HEAD_SIZE; ++d) {
            score += static_cast<Tcompute>(q_vec[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= static_cast<Tcompute>(scale);
        if (alibi_slope != 0.0f) {
            score += static_cast<Tcompute>(alibi_slope * static_cast<float>(t - causal_limit));
        }
        if (score > max_score) {
            max_score = score;
        }
    }

    Tcompute sum_exp = 0;
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t page = t / pbs;
        const size_t off = t - page * pbs;
        const ptrdiff_t phys = static_cast<ptrdiff_t>(block_table[page]);
        const Tdata *k_vec = k_cache_ + static_cast<int64_t>(phys) * k_batch_stride + static_cast<int64_t>(off) * k_row_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;

        Tcompute score = 0;
        for (size_t d = 0; d < HEAD_SIZE; ++d) {
            score += static_cast<Tcompute>(q_vec[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= static_cast<Tcompute>(scale);
        if (alibi_slope != 0.0f) {
            score += static_cast<Tcompute>(alibi_slope * static_cast<float>(t - causal_limit));
        }
        sum_exp += static_cast<Tcompute>(expf(static_cast<float>(score - max_score)));
    }

    const Tcompute inv_sum = static_cast<Tcompute>(1.0f) / (sum_exp + static_cast<Tcompute>(1e-6f));
    Tcompute acc = 0;
    for (size_t t = 0; t <= causal_limit; ++t) {
        const size_t page = t / pbs;
        const size_t off = t - page * pbs;
        const ptrdiff_t phys = static_cast<ptrdiff_t>(block_table[page]);
        const Tdata *k_vec = k_cache_ + static_cast<int64_t>(phys) * k_batch_stride + static_cast<int64_t>(off) * k_row_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;

        Tcompute score = 0;
        for (size_t d = 0; d < HEAD_SIZE; ++d) {
            score += static_cast<Tcompute>(q_vec[d]) * static_cast<Tcompute>(k_vec[d]);
        }
        score *= static_cast<Tcompute>(scale);
        if (alibi_slope != 0.0f) {
            score += static_cast<Tcompute>(alibi_slope * static_cast<float>(t - causal_limit));
        }
        const Tcompute prob = static_cast<Tcompute>(expf(static_cast<float>(score - max_score))) * inv_sum;

        const Tdata *v_vec = v_cache_ + static_cast<int64_t>(phys) * v_batch_stride + static_cast<int64_t>(off) * v_row_stride + static_cast<int64_t>(kv_head_idx) * v_head_stride;
        acc += prob * static_cast<Tcompute>(v_vec[dim_idx]);
    }

    out_ptr[dim_idx] = static_cast<Tdata>(acc);
}

template <typename Tindex, typename Tdata, int HEAD_SIZE, int BLOCK_M, int BLOCK_N>
__device__ void PagedAttentionPrefillWarpCtaKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cu_seqlens_q_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {

    static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128, "Only head_size 64/128 supported in v0.4.");
    static_assert(BLOCK_M > 0 && BLOCK_M <= 16, "BLOCK_M must be small (warp-per-query design).");
    static_assert(BLOCK_N == 64 || BLOCK_N == 128, "BLOCK_N must be 64/128 in v0.4.");

    constexpr int kWarpSize = 32;
    constexpr int DIMS_PER_THREAD = HEAD_SIZE / kWarpSize;
    static_assert(HEAD_SIZE % kWarpSize == 0, "HEAD_SIZE must be divisible by 32.");

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    if (warp_id >= BLOCK_M) {
        return;
    }

    const int head_idx = static_cast<int>(blockIdx.x);
    const int seq_idx = static_cast<int>(blockIdx.y);
    const int m_block = static_cast<int>(blockIdx.z);

    const Tindex q_start = cu_seqlens_q_[seq_idx];
    const Tindex q_end = cu_seqlens_q_[seq_idx + 1];
    const int q_len = static_cast<int>(q_end - q_start);
    if (q_len <= 0) {
        return;
    }

    const int m_start = m_block * BLOCK_M;
    const int q_token_local = m_start + warp_id;
    // IMPORTANT: do not early-return for a subset of warps in this CTA because we use __syncthreads()
    // later. Tail tiles are handled by masking inactive warps.
    if (m_start >= q_len) {
        return; // uniform across the CTA
    }
    const bool is_active = (q_token_local < q_len);

    const int64_t kv_len_total_i64 = total_kv_lens_[seq_idx];
    const int kv_len_total = static_cast<int>(kv_len_total_i64);
    // history_len = total_kv_len - q_len (KV already includes current q tokens).
    const int history_len = kv_len_total - q_len;
    const int allowed_k_len = is_active ? (history_len + q_token_local + 1) : 0;

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    int64_t q_token = q_start;
    if (is_active) {
        q_token += static_cast<int64_t>(q_token_local);
    }

    const Tindex *block_table = block_tables_ + static_cast<int64_t>(seq_idx) * static_cast<int64_t>(block_table_batch_stride);

    const Tdata *q_ptr = nullptr;
    Tdata *out_ptr = nullptr;
    if (is_active) {
        q_ptr = q_ + q_token * q_stride + static_cast<int64_t>(head_idx) * q_head_stride;
        out_ptr = out_ + q_token * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;
    }

    float q_reg[DIMS_PER_THREAD];
    float acc[DIMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        q_reg[i] = is_active ? static_cast<float>(q_ptr[dim]) : 0.0f;
        acc[i] = 0.0f;
    }

#if defined(__CUDA_ARCH__)
    float2 q_reg2[DIMS_PER_THREAD / 2];
#pragma unroll
    for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
        q_reg2[j] = make_float2(q_reg[j * 2 + 0], q_reg[j * 2 + 1]);
    }
#endif

    float m = -INFINITY;
    float l = 0.0f;

    // For this CTA, we only need to scan up to the max allowed k among active warps.
    const int max_q_in_tile = min(m_start + BLOCK_M, q_len);
    const int max_allowed_k_len = min(history_len + max_q_in_tile, kv_len_total);

    __shared__ int32_t s_phys[BLOCK_N];
    __shared__ int32_t s_off[BLOCK_N];
    // Ensure shared-memory tiles are aligned for half2/bfloat162 vector loads.
    __shared__ __align__(16) Tdata s_k[BLOCK_N * HEAD_SIZE];
    __shared__ __align__(16) Tdata s_v[BLOCK_N * HEAD_SIZE];

    const int pbs = static_cast<int>(page_block_size);

    for (int k_base = 0; k_base < max_allowed_k_len; k_base += BLOCK_N) {
        const int tile_n = min(BLOCK_N, max_allowed_k_len - k_base);

        // Precompute page mapping once per token in the tile.
        for (int t = threadIdx.x; t < tile_n; t += blockDim.x) {
            const int kpos = k_base + t;
            const int page = (pbs == 256) ? (kpos >> 8) : (kpos / pbs);
            const int off = (pbs == 256) ? (kpos & 255) : (kpos - page * pbs);
            const int32_t phys = static_cast<int32_t>(block_table[page]);
            s_phys[t] = phys;
            s_off[t] = off;
        }
        __syncthreads();

        // Load K/V tile into shared memory (contiguous in head_dim).
        const int tile_elems = tile_n * HEAD_SIZE;
        for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            const int t = idx / HEAD_SIZE;
            const int dim = idx - t * HEAD_SIZE;
            const int32_t phys = s_phys[t];
            const int32_t off = s_off[t];
            const Tdata *k_base_ptr = k_cache_ + static_cast<int64_t>(phys) * k_batch_stride + static_cast<int64_t>(off) * k_row_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;
            const Tdata *v_base_ptr = v_cache_ + static_cast<int64_t>(phys) * v_batch_stride + static_cast<int64_t>(off) * v_row_stride + static_cast<int64_t>(kv_head_idx) * v_head_stride;
            s_k[t * HEAD_SIZE + dim] = k_base_ptr[dim];
            s_v[t * HEAD_SIZE + dim] = v_base_ptr[dim];
        }
        __syncthreads();

        // Each warp processes one query token and scans the K/V tile.
        for (int t = 0; t < tile_n; ++t) {
            const int kpos = k_base + t;
            if (kpos >= allowed_k_len) {
                break;
            }
            const Tdata *k_ptr = s_k + t * HEAD_SIZE;
            const Tdata *v_ptr = s_v + t * HEAD_SIZE;

            float qk = 0.0f;
#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<Tdata, half>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const half2 *k2 = reinterpret_cast<const half2 *>(k_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 qf = q_reg2[j];
                    const float2 kf = __half22float2(k2[j]);
                    qk += qf.x * kf.x + qf.y * kf.y;
                }
            } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const __nv_bfloat162 *k2 = reinterpret_cast<const __nv_bfloat162 *>(k_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 qf = q_reg2[j];
                    const float2 kf = __bfloat1622float2(k2[j]);
                    qk += qf.x * kf.x + qf.y * kf.y;
                }
            } else
#endif
#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    const int dim = lane * DIMS_PER_THREAD + i;
                    qk += q_reg[i] * static_cast<float>(k_ptr[dim]);
                }

            qk = op::paged_attention::cuda::warpReduceSum(qk);

            float alpha = 1.0f;
            float beta = 0.0f;
            if (lane == 0) {
                float score = qk * scale_log2;
                if (alibi_slope != 0.0f) {
                    // Causal prefill: last position is (allowed_k_len - 1) for this query.
                    score += (alibi_slope * static_cast<float>(kpos - (allowed_k_len - 1))) * kLog2e;
                }
                const float m_new = fmaxf(m, score);
                alpha = exp2f(m - m_new);
                beta = exp2f(score - m_new);
                l = l * alpha + beta;
                m = m_new;
            }
            alpha = op::paged_attention::cuda::warpBroadcast(alpha, 0);
            beta = op::paged_attention::cuda::warpBroadcast(beta, 0);

#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<Tdata, half>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const half2 *v2 = reinterpret_cast<const half2 *>(v_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 vf = __half22float2(v2[j]);
                    acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
                    acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
                }
            } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const __nv_bfloat162 *v2 = reinterpret_cast<const __nv_bfloat162 *>(v_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 vf = __bfloat1622float2(v2[j]);
                    acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
                    acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
                }
            } else
#endif
            {
#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    const int dim = lane * DIMS_PER_THREAD + i;
                    const float v_val = static_cast<float>(v_ptr[dim]);
                    acc[i] = acc[i] * alpha + beta * v_val;
                }
            }
        }

        __syncthreads();
    }

    float inv_l = 0.0f;
    if (lane == 0) {
        inv_l = 1.0f / (l + 1e-6f);
    }
    inv_l = op::paged_attention::cuda::warpBroadcast(inv_l, 0);

#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        const float out_val = acc[i] * inv_l;
        if (!is_active) {
            continue;
        }
        if constexpr (std::is_same_v<Tdata, half>) {
            out_ptr[dim] = __float2half_rn(out_val);
        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
            out_ptr[dim] = __float2bfloat16_rn(out_val);
        } else {
            out_ptr[dim] = static_cast<Tdata>(out_val);
        }
    }
}

// Pipelined CTA kernel (FA2-style): stage K/V loads with cp.async and overlap global->shared
// copies with compute.
//
// Design notes:
// - Keep shared memory <= 48KB for compatibility with multi-arch builds that include SM75.
// - Iterate by paged blocks (logical pages) so each tile stays within one physical block and
//   avoids per-token (page, off) mapping arrays in shared memory.
// - One warp computes one query token (same as warpcta kernels). Warps with shorter causal
//   limits simply mask the tail tokens but still participate in CTA-wide barriers.
template <typename Tindex, typename Tdata, int HEAD_SIZE, int BLOCK_M, int TOKENS_PER_TILE, int STAGES>
__device__ void PagedAttentionPrefillWarpCtaKernelPipelined(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cu_seqlens_q_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {

    static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128, "Only head_size 64/128 supported in v0.4.");
    static_assert(BLOCK_M > 0 && BLOCK_M <= 16, "BLOCK_M must be <= 16.");
    static_assert(TOKENS_PER_TILE == 32, "Pipelined CTA kernel currently assumes TOKENS_PER_TILE == 32.");
    static_assert(STAGES >= 2 && STAGES <= 3, "STAGES must be 2 or 3.");
    static_assert(sizeof(Tdata) == 2, "Pipelined CTA kernel supports only fp16/bf16.");

    constexpr int kWarpSize = 32;
    static_assert(HEAD_SIZE % kWarpSize == 0, "HEAD_SIZE must be divisible by 32.");
    constexpr int DIMS_PER_THREAD = HEAD_SIZE / kWarpSize;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    if (warp_id >= BLOCK_M) {
        return;
    }

    const int head_idx = static_cast<int>(blockIdx.x);
    const int seq_idx = static_cast<int>(blockIdx.y);
    const int m_block = static_cast<int>(blockIdx.z);

    const Tindex q_start = cu_seqlens_q_[seq_idx];
    const Tindex q_end = cu_seqlens_q_[seq_idx + 1];
    const int q_len = static_cast<int>(q_end - q_start);
    if (q_len <= 0) {
        return;
    }

    const int m_start = m_block * BLOCK_M;
    const int q_token_local = m_start + warp_id;
    // Uniform return for empty tail CTAs (avoid deadlock with __syncthreads).
    if (m_start >= q_len) {
        return;
    }
    const bool is_active = (q_token_local < q_len);

    const int kv_len_total = static_cast<int>(total_kv_lens_[seq_idx]);
    const int history_len = kv_len_total - q_len;
    const int allowed_k_len = is_active ? (history_len + q_token_local + 1) : 0;

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    int64_t q_token = q_start;
    if (is_active) {
        q_token += static_cast<int64_t>(q_token_local);
    }

    const Tindex *block_table = block_tables_ + static_cast<int64_t>(seq_idx) * static_cast<int64_t>(block_table_batch_stride);

    const Tdata *q_ptr = nullptr;
    Tdata *out_ptr = nullptr;
    if (is_active) {
        q_ptr = q_ + q_token * q_stride + static_cast<int64_t>(head_idx) * q_head_stride;
        out_ptr = out_ + q_token * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;
    }

    float q_reg[DIMS_PER_THREAD];
    float acc[DIMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        q_reg[i] = is_active ? static_cast<float>(q_ptr[dim]) : 0.0f;
        acc[i] = 0.0f;
    }

#if defined(__CUDA_ARCH__)
    float2 q_reg2[DIMS_PER_THREAD / 2];
#pragma unroll
    for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
        q_reg2[j] = make_float2(q_reg[j * 2 + 0], q_reg[j * 2 + 1]);
    }
#endif

    float m = -INFINITY;
    float l = 0.0f;

    // For this CTA, scan KV up to the max causal limit among active warps.
    const int max_q_in_tile = min(m_start + BLOCK_M, q_len);
    const int max_allowed_k_len = min(history_len + max_q_in_tile, kv_len_total);
    if (max_allowed_k_len <= 0) {
        // Nothing to attend to (should be rare). Produce zeros.
        if (is_active) {
#pragma unroll
            for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                const int dim = lane * DIMS_PER_THREAD + i;
                out_ptr[dim] = Tdata{};
            }
        }
        return;
    }

    // cp.async uses 16B chunks; for fp16/bf16 that's 8 elements.
    constexpr int CHUNK_ELEMS = 8;
    constexpr int CHUNKS = HEAD_SIZE / CHUNK_ELEMS;
    constexpr int LOADS_PER_TILE = CHUNKS * TOKENS_PER_TILE;

    // Multi-stage pipeline buffers.
    __shared__ __align__(16) Tdata sh_k[STAGES][TOKENS_PER_TILE][HEAD_SIZE];
    __shared__ __align__(16) Tdata sh_v[STAGES][TOKENS_PER_TILE][HEAD_SIZE];
    // Per-warp scratch for tile-wise softmax (scores over TOKENS_PER_TILE).
    // We keep scores in shared so each lane can load its token score (lane -> token index),
    // then weights are broadcast via warp shuffles to avoid extra shared-memory traffic.
    __shared__ float sh_scores[BLOCK_M][TOKENS_PER_TILE];
    // Store Q in shared (per warp). This enables more tile-level parallelism in score
    // computation without expensive cross-lane shuffles of Q registers.
    __shared__ __align__(16) Tdata sh_q[BLOCK_M][HEAD_SIZE];

    const int pbs = static_cast<int>(page_block_size);
    const int tid = threadIdx.x;

    // Populate per-warp Q shared tile once.
#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        sh_q[warp_id][dim] = is_active ? q_ptr[dim] : Tdata{};
    }
    __syncwarp();

    int t_base = 0;
    for (int logical_block = 0; t_base < max_allowed_k_len; ++logical_block, t_base += pbs) {
        const int physical_block = static_cast<int>(block_table[logical_block]);

        const Tdata *k_base = k_cache_ + static_cast<int64_t>(physical_block) * k_batch_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;
        const Tdata *v_base = v_cache_ + static_cast<int64_t>(physical_block) * v_batch_stride + static_cast<int64_t>(kv_head_idx) * v_head_stride;

        const int token_end = min(pbs, max_allowed_k_len - t_base);
        const int num_tiles = (token_end + TOKENS_PER_TILE - 1) / TOKENS_PER_TILE;
        if (num_tiles <= 0) {
            continue;
        }

        int pending_groups = 0;
        const int preload = min(STAGES, num_tiles);
        for (int ti = 0; ti < preload; ++ti) {
            const int token_in_block = ti * TOKENS_PER_TILE;
            const int tile_n = min(TOKENS_PER_TILE, token_end - token_in_block);
            for (int li = tid; li < LOADS_PER_TILE; li += blockDim.x) {
                const int tok = li / CHUNKS;
                const int chunk = li - tok * CHUNKS;
                const int off = chunk * CHUNK_ELEMS;
                if (tok < tile_n) {
                    const Tdata *k_src = k_base + static_cast<int64_t>(token_in_block + tok) * k_row_stride + off;
                    const Tdata *v_src = v_base + static_cast<int64_t>(token_in_block + tok) * v_row_stride + off;
                    op::paged_attention::cuda::cpAsyncCaSharedGlobal16(&sh_k[ti][tok][off], k_src);
                    op::paged_attention::cuda::cpAsyncCaSharedGlobal16(&sh_v[ti][tok][off], v_src);
                } else {
                    reinterpret_cast<uint4 *>(&sh_k[ti][tok][off])[0] = make_uint4(0, 0, 0, 0);
                    reinterpret_cast<uint4 *>(&sh_v[ti][tok][off])[0] = make_uint4(0, 0, 0, 0);
                }
            }
            op::paged_attention::cuda::cpAsyncCommit();
            ++pending_groups;
        }

        int desired_pending = pending_groups - 1;
        if (desired_pending < 0) {
            desired_pending = 0;
        }
        if (desired_pending > (STAGES - 1)) {
            desired_pending = (STAGES - 1);
        }
        op::paged_attention::cuda::cpAsyncWaitGroupRt(desired_pending);
        pending_groups = desired_pending;
        __syncthreads();

        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            const int buf = tile_idx % STAGES;
            const int token_in_block = tile_idx * TOKENS_PER_TILE;
            const int tile_n = min(TOKENS_PER_TILE, token_end - token_in_block);

            const int global_k_base = t_base + token_in_block;
            // Tile-wise online softmax (more FA2-like than per-token update):
            // 1) Compute scores for this tile (masked to each warp's causal limit).
            // 2) Compute tile max + sumexp.
            // 3) Accumulate weighted V for the tile.
            // 4) Merge into running (m, l, acc) in a numerically stable way.
            //
            // NOTE: this does not yet implement MMA / full tile-level GEMM; it mainly reduces
            // the serial (lane0) online-softmax update frequency from per-token to per-tile.
            float alpha = 1.0f;
            float beta = 0.0f;
            float tile_sumexp = 0.0f;
            float tile_m = -INFINITY;

            if (allowed_k_len > 0) {
                // 1) scores
                // Increase tile-level parallelism vs the previous per-token loop:
                // split the warp into 4 groups of 8 lanes; each group computes one token score in parallel.
                constexpr int LANES_PER_GROUP = 8;
                constexpr int GROUPS_PER_WARP = 4;
                constexpr int DIMS_PER_GROUP_LANE = HEAD_SIZE / LANES_PER_GROUP;
                static_assert(HEAD_SIZE % LANES_PER_GROUP == 0, "HEAD_SIZE must be divisible by 8.");

                const int group_id = lane / LANES_PER_GROUP;     // [0..3]
                const int lane_g = lane & (LANES_PER_GROUP - 1); // [0..7]
                const unsigned int group_mask = 0xFFu << (group_id * LANES_PER_GROUP);

                for (int j_base = 0; j_base < TOKENS_PER_TILE; j_base += GROUPS_PER_WARP) {
                    const int j = j_base + group_id; // token index in [0..31]
                    const int kpos = global_k_base + j;

                    const bool token_in_tile = (j < tile_n);
                    const bool token_unmasked = token_in_tile && (kpos < allowed_k_len);

                    float qk_part = 0.0f;
                    if (token_unmasked) {
                        const Tdata *k_ptr = &sh_k[buf][j][0];
                        const int dim_base = lane_g * DIMS_PER_GROUP_LANE;
#if defined(__CUDA_ARCH__)
                        if constexpr (std::is_same_v<Tdata, half>) {
                            const half2 *q2 = reinterpret_cast<const half2 *>(&sh_q[warp_id][dim_base]);
                            const half2 *k2 = reinterpret_cast<const half2 *>(k_ptr + dim_base);
#pragma unroll
                            for (int t = 0; t < DIMS_PER_GROUP_LANE / 2; ++t) {
                                const float2 qf = __half22float2(q2[t]);
                                const float2 kf = __half22float2(k2[t]);
                                qk_part += qf.x * kf.x + qf.y * kf.y;
                            }
                        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                            const __nv_bfloat162 *q2 = reinterpret_cast<const __nv_bfloat162 *>(&sh_q[warp_id][dim_base]);
                            const __nv_bfloat162 *k2 = reinterpret_cast<const __nv_bfloat162 *>(k_ptr + dim_base);
#pragma unroll
                            for (int t = 0; t < DIMS_PER_GROUP_LANE / 2; ++t) {
                                const float2 qf = __bfloat1622float2(q2[t]);
                                const float2 kf = __bfloat1622float2(k2[t]);
                                qk_part += qf.x * kf.x + qf.y * kf.y;
                            }
                        } else
#endif
                        {
#pragma unroll
                            for (int t = 0; t < DIMS_PER_GROUP_LANE; ++t) {
                                qk_part += static_cast<float>(sh_q[warp_id][dim_base + t]) * static_cast<float>(k_ptr[dim_base + t]);
                            }
                        }
                    }

                    // Reduce within 8-lane group.
                    for (int offset = LANES_PER_GROUP / 2; offset > 0; offset >>= 1) {
                        qk_part += __shfl_down_sync(group_mask, qk_part, offset, LANES_PER_GROUP);
                    }

                    if (lane_g == 0) {
                        float score = -INFINITY;
                        if (token_unmasked) {
                            score = qk_part * scale_log2;
                            if (alibi_slope != 0.0f) {
                                const int causal_limit = allowed_k_len - 1;
                                score += (alibi_slope * static_cast<float>(kpos - causal_limit)) * kLog2e;
                            }
                        }
                        sh_scores[warp_id][j] = score;
                    }
                }
                __syncwarp();

                // 2) tile max + sumexp (lane t corresponds to token t within the tile)
                const float score_lane = (lane < tile_n) ? sh_scores[warp_id][lane] : -INFINITY;
                float tile_m_tmp = op::paged_attention::cuda::warpReduceMax(score_lane);
                tile_m_tmp = __shfl_sync(0xffffffff, tile_m_tmp, 0);
                tile_m = tile_m_tmp;

                float w_lane = 0.0f;
                if (lane < tile_n && tile_m != -INFINITY) {
                    w_lane = exp2f(score_lane - tile_m);
                }
                float sumexp_tmp = op::paged_attention::cuda::warpReduceSum(w_lane);
                sumexp_tmp = __shfl_sync(0xffffffff, sumexp_tmp, 0);
                tile_sumexp = sumexp_tmp;

                // 3) weighted V for this tile (per lane owns HEAD_SIZE/32 dims)
                float acc_tile[DIMS_PER_THREAD];
#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    acc_tile[i] = 0.0f;
                }

                if (tile_sumexp > 0.0f) {
                    for (int j = 0; j < tile_n; ++j) {
                        // Broadcast weight for token j from lane j.
                        const float wj = __shfl_sync(0xffffffff, w_lane, j);
                        const Tdata *v_ptr = &sh_v[buf][j][0];
#if defined(__CUDA_ARCH__)
                        if constexpr (std::is_same_v<Tdata, half>) {
                            const int dim_base = lane * DIMS_PER_THREAD;
                            const half2 *v2 = reinterpret_cast<const half2 *>(v_ptr + dim_base);
#pragma unroll
                            for (int jj = 0; jj < DIMS_PER_THREAD / 2; ++jj) {
                                const float2 vf = __half22float2(v2[jj]);
                                acc_tile[jj * 2 + 0] += wj * vf.x;
                                acc_tile[jj * 2 + 1] += wj * vf.y;
                            }
                        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                            const int dim_base = lane * DIMS_PER_THREAD;
                            const __nv_bfloat162 *v2 = reinterpret_cast<const __nv_bfloat162 *>(v_ptr + dim_base);
#pragma unroll
                            for (int jj = 0; jj < DIMS_PER_THREAD / 2; ++jj) {
                                const float2 vf = __bfloat1622float2(v2[jj]);
                                acc_tile[jj * 2 + 0] += wj * vf.x;
                                acc_tile[jj * 2 + 1] += wj * vf.y;
                            }
                        } else
#endif
                        {
#pragma unroll
                            for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                                const int dim = lane * DIMS_PER_THREAD + i;
                                acc_tile[i] += wj * static_cast<float>(v_ptr[dim]);
                            }
                        }
                    }
                }

                // 4) merge tile into running (m, l, acc)
                if (lane == 0) {
                    if (tile_sumexp > 0.0f && tile_m != -INFINITY) {
                        const float m_new = fmaxf(m, tile_m);
                        alpha = exp2f(m - m_new);
                        beta = exp2f(tile_m - m_new);
                        l = l * alpha + tile_sumexp * beta;
                        m = m_new;
                    } else {
                        alpha = 1.0f;
                        beta = 0.0f;
                    }
                }
                alpha = __shfl_sync(0xffffffff, alpha, 0);
                beta = __shfl_sync(0xffffffff, beta, 0);

#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    acc[i] = acc[i] * alpha + beta * acc_tile[i];
                }
            }

            // IMPORTANT: warps in this CTA can have different allowed_k_len (due to causal mask + history),
            // so they may finish the token loop at different times. We must not start prefetching into
            // the circular shared-memory buffer until all warps finish consuming the current tile.
            __syncthreads();

            // Prefetch the tile that will reuse this buffer (STAGES steps ahead).
            const int prefetch_tile = tile_idx + STAGES;
            if (prefetch_tile < num_tiles) {
                const int token_prefetch = prefetch_tile * TOKENS_PER_TILE;
                const int prefetch_n = min(TOKENS_PER_TILE, token_end - token_prefetch);
                for (int li = tid; li < LOADS_PER_TILE; li += blockDim.x) {
                    const int tok = li / CHUNKS;
                    const int chunk = li - tok * CHUNKS;
                    const int off = chunk * CHUNK_ELEMS;
                    if (tok < prefetch_n) {
                        const Tdata *k_src = k_base + static_cast<int64_t>(token_prefetch + tok) * k_row_stride + off;
                        const Tdata *v_src = v_base + static_cast<int64_t>(token_prefetch + tok) * v_row_stride + off;
                        op::paged_attention::cuda::cpAsyncCaSharedGlobal16(&sh_k[buf][tok][off], k_src);
                        op::paged_attention::cuda::cpAsyncCaSharedGlobal16(&sh_v[buf][tok][off], v_src);
                    } else {
                        reinterpret_cast<uint4 *>(&sh_k[buf][tok][off])[0] = make_uint4(0, 0, 0, 0);
                        reinterpret_cast<uint4 *>(&sh_v[buf][tok][off])[0] = make_uint4(0, 0, 0, 0);
                    }
                }
                op::paged_attention::cuda::cpAsyncCommit();
                ++pending_groups;
            }

            if (tile_idx + 1 < num_tiles) {
                int desired_pending2 = pending_groups - 1;
                if (desired_pending2 < 0) {
                    desired_pending2 = 0;
                }
                if (desired_pending2 > (STAGES - 1)) {
                    desired_pending2 = (STAGES - 1);
                }
                op::paged_attention::cuda::cpAsyncWaitGroupRt(desired_pending2);
                pending_groups = desired_pending2;
                __syncthreads();
            }
        }

        op::paged_attention::cuda::cpAsyncWaitAll();
        __syncthreads();
    }

    float inv_l = 0.0f;
    if (lane == 0) {
        inv_l = 1.0f / (l + 1e-6f);
    }
    inv_l = op::paged_attention::cuda::warpBroadcast(inv_l, 0);

#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        const float out_val = acc[i] * inv_l;
        if (!is_active) {
            continue;
        }
        if constexpr (std::is_same_v<Tdata, half>) {
            out_ptr[dim] = __float2half_rn(out_val);
        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
            out_ptr[dim] = __float2bfloat16_rn(out_val);
        } else {
            out_ptr[dim] = static_cast<Tdata>(out_val);
        }
    }
}

// Split-KV prefill (FA2-style): each split scans a shard of KV and writes partial (m, l, acc)
// to workspace. A separate combine kernel merges splits into the final output.
//
// Notes:
// - Implemented for the pipelined CTA kernel family (warpcta8pipe). We split by logical paged blocks.
// - Each warp still applies its own causal limit (allowed_k_len) so correctness is preserved.
template <typename Tindex, typename Tdata, int HEAD_SIZE, int BLOCK_M, int TOKENS_PER_TILE, int STAGES>
__device__ void PagedAttentionPrefillWarpCtaKernelPipelinedSplitKv(
    float *partial_acc, // [num_splits, total_q_tokens, num_heads, head_size]
    float *partial_m,   // [num_splits, total_q_tokens, num_heads]
    float *partial_l,   // [num_splits, total_q_tokens, num_heads]
    int split_idx,
    int num_splits,
    int m_block,
    size_t total_q_tokens,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cu_seqlens_q_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride) {

    (void)max_num_blocks_per_seq;

    static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128, "Only head_size 64/128 supported in v0.4.");
    static_assert(BLOCK_M > 0 && BLOCK_M <= 16, "BLOCK_M must be <= 16.");
    static_assert(TOKENS_PER_TILE == 32, "Split-KV prefill assumes TOKENS_PER_TILE == 32.");
    static_assert(STAGES >= 2 && STAGES <= 3, "STAGES must be 2 or 3.");
    static_assert(sizeof(Tdata) == 2, "Split-KV prefill supports only fp16/bf16.");

    constexpr int kWarpSize = 32;
    static_assert(HEAD_SIZE % kWarpSize == 0, "HEAD_SIZE must be divisible by 32.");
    constexpr int DIMS_PER_THREAD = HEAD_SIZE / kWarpSize;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    if (warp_id >= BLOCK_M) {
        return;
    }

    const int head_idx = static_cast<int>(blockIdx.x);
    const int seq_idx = static_cast<int>(blockIdx.y);

    const Tindex q_start = cu_seqlens_q_[seq_idx];
    const Tindex q_end = cu_seqlens_q_[seq_idx + 1];
    const int q_len = static_cast<int>(q_end - q_start);
    if (q_len <= 0) {
        return;
    }

    const int m_start = m_block * BLOCK_M;
    const int q_token_local = m_start + warp_id;
    if (m_start >= q_len) {
        return; // uniform
    }
    const bool is_active = (q_token_local < q_len);

    const int kv_len_total = static_cast<int>(total_kv_lens_[seq_idx]);
    const int history_len = kv_len_total - q_len;
    const int allowed_k_len = is_active ? (history_len + q_token_local + 1) : 0;

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    int64_t q_token = q_start;
    if (is_active) {
        q_token += static_cast<int64_t>(q_token_local);
    }

    const size_t n = total_q_tokens * static_cast<size_t>(num_heads);
    size_t base = 0;
    if (is_active) {
        base = static_cast<size_t>(q_token) * static_cast<size_t>(num_heads) + static_cast<size_t>(head_idx);
    }

    const Tindex *block_table = block_tables_ + static_cast<int64_t>(seq_idx) * static_cast<int64_t>(block_table_batch_stride);
    const Tdata *q_ptr = nullptr;
    if (is_active) {
        q_ptr = q_ + q_token * q_stride + static_cast<int64_t>(head_idx) * q_head_stride;
    }

    float q_reg[DIMS_PER_THREAD];
    float acc[DIMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        q_reg[i] = is_active ? static_cast<float>(q_ptr[dim]) : 0.0f;
        acc[i] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;

    const int max_q_in_tile = min(m_start + BLOCK_M, q_len);
    const int max_allowed_k_len = min(history_len + max_q_in_tile, kv_len_total);
    if (max_allowed_k_len <= 0) {
        if (is_active) {
            const size_t idx = static_cast<size_t>(split_idx) * n + base;
            if (lane == 0) {
                partial_m[idx] = -INFINITY;
                partial_l[idx] = 0.0f;
            }
#pragma unroll
            for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                const int dim = lane * DIMS_PER_THREAD + i;
                partial_acc[idx * HEAD_SIZE + dim] = 0.0f;
            }
        }
        return;
    }

    const int pbs = static_cast<int>(page_block_size);
    const int num_blocks_total = (max_allowed_k_len + pbs - 1) / pbs;
    const int blocks_per_split = (num_blocks_total + num_splits - 1) / num_splits;
    const int start_block = split_idx * blocks_per_split;
    const int end_block = min(num_blocks_total, start_block + blocks_per_split);
    if (start_block >= end_block) {
        if (is_active) {
            const size_t idx = static_cast<size_t>(split_idx) * n + base;
            if (lane == 0) {
                partial_m[idx] = -INFINITY;
                partial_l[idx] = 0.0f;
            }
#pragma unroll
            for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                const int dim = lane * DIMS_PER_THREAD + i;
                partial_acc[idx * HEAD_SIZE + dim] = 0.0f;
            }
        }
        return;
    }

    const int max_allowed_k_len_split = min(max_allowed_k_len, end_block * pbs);

    constexpr int CHUNK_ELEMS = 8;
    constexpr int CHUNKS = HEAD_SIZE / CHUNK_ELEMS;
    constexpr int LOADS_PER_TILE = CHUNKS * TOKENS_PER_TILE;

    __shared__ __align__(16) Tdata sh_k[STAGES][TOKENS_PER_TILE][HEAD_SIZE];
    __shared__ __align__(16) Tdata sh_v[STAGES][TOKENS_PER_TILE][HEAD_SIZE];
    __shared__ float sh_scores[BLOCK_M][TOKENS_PER_TILE];

    const int tid = threadIdx.x;

    int t_base = start_block * pbs;
    for (int logical_block = start_block; t_base < max_allowed_k_len_split; ++logical_block, t_base += pbs) {
        const int physical_block = static_cast<int>(block_table[logical_block]);

        const Tdata *k_base = k_cache_ + static_cast<int64_t>(physical_block) * k_batch_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;
        const Tdata *v_base = v_cache_ + static_cast<int64_t>(physical_block) * v_batch_stride + static_cast<int64_t>(kv_head_idx) * v_head_stride;

        const int token_end = min(pbs, max_allowed_k_len_split - t_base);
        const int num_tiles = (token_end + TOKENS_PER_TILE - 1) / TOKENS_PER_TILE;
        if (num_tiles <= 0) {
            continue;
        }

        int pending_groups = 0;
        const int preload = min(STAGES, num_tiles);
        for (int ti = 0; ti < preload; ++ti) {
            const int token_in_block = ti * TOKENS_PER_TILE;
            const int tile_n = min(TOKENS_PER_TILE, token_end - token_in_block);
            for (int li = tid; li < LOADS_PER_TILE; li += blockDim.x) {
                const int tok = li / CHUNKS;
                const int chunk = li - tok * CHUNKS;
                const int off = chunk * CHUNK_ELEMS;
                if (tok < tile_n) {
                    const Tdata *k_src = k_base + static_cast<int64_t>(token_in_block + tok) * k_row_stride + off;
                    const Tdata *v_src = v_base + static_cast<int64_t>(token_in_block + tok) * v_row_stride + off;
                    op::paged_attention::cuda::cpAsyncCaSharedGlobal16(&sh_k[ti][tok][off], k_src);
                    op::paged_attention::cuda::cpAsyncCaSharedGlobal16(&sh_v[ti][tok][off], v_src);
                } else {
                    reinterpret_cast<uint4 *>(&sh_k[ti][tok][off])[0] = make_uint4(0, 0, 0, 0);
                    reinterpret_cast<uint4 *>(&sh_v[ti][tok][off])[0] = make_uint4(0, 0, 0, 0);
                }
            }
            op::paged_attention::cuda::cpAsyncCommit();
            ++pending_groups;
        }

        int desired_pending = pending_groups - 1;
        if (desired_pending < 0) {
            desired_pending = 0;
        }
        if (desired_pending > (STAGES - 1)) {
            desired_pending = (STAGES - 1);
        }
        op::paged_attention::cuda::cpAsyncWaitGroupRt(desired_pending);
        pending_groups = desired_pending;
        __syncthreads();

        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            const int buf = tile_idx % STAGES;
            const int token_in_block = tile_idx * TOKENS_PER_TILE;
            const int tile_n = min(TOKENS_PER_TILE, token_end - token_in_block);
            const int global_k_base = t_base + token_in_block;

            float alpha = 1.0f;
            float beta = 0.0f;
            float tile_sumexp = 0.0f;
            float tile_m = -INFINITY;
            float w_lane = 0.0f;

            if (allowed_k_len > 0) {
                // 1) scores
                for (int j = 0; j < tile_n; ++j) {
                    const int kpos = global_k_base + j;
                    const bool token_unmasked = (kpos < allowed_k_len);
                    float qk = 0.0f;
                    if (token_unmasked) {
                        const Tdata *k_ptr = &sh_k[buf][j][0];
#if defined(__CUDA_ARCH__)
                        if constexpr (std::is_same_v<Tdata, half>) {
                            const int dim_base = lane * DIMS_PER_THREAD;
                            const half2 *q2 = reinterpret_cast<const half2 *>(q_ptr + dim_base);
                            const half2 *k2 = reinterpret_cast<const half2 *>(k_ptr + dim_base);
#pragma unroll
                            for (int ii = 0; ii < DIMS_PER_THREAD / 2; ++ii) {
                                const float2 qf = __half22float2(q2[ii]);
                                const float2 kf = __half22float2(k2[ii]);
                                qk = fmaf(qf.x, kf.x, qk);
                                qk = fmaf(qf.y, kf.y, qk);
                            }
                        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                            const int dim_base = lane * DIMS_PER_THREAD;
                            const __nv_bfloat162 *q2 = reinterpret_cast<const __nv_bfloat162 *>(q_ptr + dim_base);
                            const __nv_bfloat162 *k2 = reinterpret_cast<const __nv_bfloat162 *>(k_ptr + dim_base);
#pragma unroll
                            for (int ii = 0; ii < DIMS_PER_THREAD / 2; ++ii) {
                                const float2 qf = __bfloat1622float2(q2[ii]);
                                const float2 kf = __bfloat1622float2(k2[ii]);
                                qk = fmaf(qf.x, kf.x, qk);
                                qk = fmaf(qf.y, kf.y, qk);
                            }
                        } else
#endif
                        {
#pragma unroll
                            for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                                const int dim = lane * DIMS_PER_THREAD + i;
                                qk = fmaf(q_reg[i], static_cast<float>(k_ptr[dim]), qk);
                            }
                        }
                    }
                    qk = op::paged_attention::cuda::warpReduceSum(qk);
                    if (lane == 0) {
                        float score = token_unmasked ? (qk * scale_log2) : -INFINITY;
                        if (token_unmasked && alibi_slope != 0.0f) {
                            const int causal_limit = allowed_k_len - 1;
                            score += (alibi_slope * static_cast<float>(kpos - causal_limit)) * kLog2e;
                        }
                        sh_scores[warp_id][j] = score;
                    }
                }
                __syncwarp();

                // 2) tile max / sumexp
                float max_tmp = -INFINITY;
                if (lane < tile_n) {
                    max_tmp = sh_scores[warp_id][lane];
                }
                max_tmp = op::paged_attention::cuda::warpReduceMax(max_tmp);
                max_tmp = __shfl_sync(0xffffffff, max_tmp, 0);
                tile_m = max_tmp;

                if (lane < tile_n) {
                    const float s = sh_scores[warp_id][lane];
                    w_lane = (s == -INFINITY) ? 0.0f : exp2f(s - tile_m);
                } else {
                    w_lane = 0.0f;
                }
                float sumexp_tmp = op::paged_attention::cuda::warpReduceSum(w_lane);
                sumexp_tmp = __shfl_sync(0xffffffff, sumexp_tmp, 0);
                tile_sumexp = sumexp_tmp;

                // 3) weighted V for this tile
                float acc_tile[DIMS_PER_THREAD];
#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    acc_tile[i] = 0.0f;
                }
                if (tile_sumexp > 0.0f) {
                    for (int j = 0; j < tile_n; ++j) {
                        const float wj = __shfl_sync(0xffffffff, w_lane, j);
                        const Tdata *v_ptr = &sh_v[buf][j][0];
#if defined(__CUDA_ARCH__)
                        if constexpr (std::is_same_v<Tdata, half>) {
                            const int dim_base = lane * DIMS_PER_THREAD;
                            const half2 *v2 = reinterpret_cast<const half2 *>(v_ptr + dim_base);
#pragma unroll
                            for (int jj = 0; jj < DIMS_PER_THREAD / 2; ++jj) {
                                const float2 vf = __half22float2(v2[jj]);
                                acc_tile[jj * 2 + 0] += wj * vf.x;
                                acc_tile[jj * 2 + 1] += wj * vf.y;
                            }
                        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                            const int dim_base = lane * DIMS_PER_THREAD;
                            const __nv_bfloat162 *v2 = reinterpret_cast<const __nv_bfloat162 *>(v_ptr + dim_base);
#pragma unroll
                            for (int jj = 0; jj < DIMS_PER_THREAD / 2; ++jj) {
                                const float2 vf = __bfloat1622float2(v2[jj]);
                                acc_tile[jj * 2 + 0] += wj * vf.x;
                                acc_tile[jj * 2 + 1] += wj * vf.y;
                            }
                        } else
#endif
                        {
#pragma unroll
                            for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                                const int dim = lane * DIMS_PER_THREAD + i;
                                acc_tile[i] += wj * static_cast<float>(v_ptr[dim]);
                            }
                        }
                    }
                }

                // 4) merge tile into running (m, l, acc)
                if (lane == 0) {
                    if (tile_sumexp > 0.0f && tile_m != -INFINITY) {
                        const float m_new = fmaxf(m, tile_m);
                        alpha = exp2f(m - m_new);
                        beta = exp2f(tile_m - m_new);
                        l = l * alpha + tile_sumexp * beta;
                        m = m_new;
                    } else {
                        alpha = 1.0f;
                        beta = 0.0f;
                    }
                }
                alpha = __shfl_sync(0xffffffff, alpha, 0);
                beta = __shfl_sync(0xffffffff, beta, 0);
#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    acc[i] = acc[i] * alpha + beta * acc_tile[i];
                }
            }

            __syncthreads();

            const int prefetch_tile = tile_idx + STAGES;
            if (prefetch_tile < num_tiles) {
                const int token_prefetch = prefetch_tile * TOKENS_PER_TILE;
                const int prefetch_n = min(TOKENS_PER_TILE, token_end - token_prefetch);
                for (int li = tid; li < LOADS_PER_TILE; li += blockDim.x) {
                    const int tok = li / CHUNKS;
                    const int chunk = li - tok * CHUNKS;
                    const int off = chunk * CHUNK_ELEMS;
                    if (tok < prefetch_n) {
                        const Tdata *k_src = k_base + static_cast<int64_t>(token_prefetch + tok) * k_row_stride + off;
                        const Tdata *v_src = v_base + static_cast<int64_t>(token_prefetch + tok) * v_row_stride + off;
                        op::paged_attention::cuda::cpAsyncCaSharedGlobal16(&sh_k[buf][tok][off], k_src);
                        op::paged_attention::cuda::cpAsyncCaSharedGlobal16(&sh_v[buf][tok][off], v_src);
                    } else {
                        reinterpret_cast<uint4 *>(&sh_k[buf][tok][off])[0] = make_uint4(0, 0, 0, 0);
                        reinterpret_cast<uint4 *>(&sh_v[buf][tok][off])[0] = make_uint4(0, 0, 0, 0);
                    }
                }
                op::paged_attention::cuda::cpAsyncCommit();
                ++pending_groups;
            }

            if (tile_idx + 1 < num_tiles) {
                int desired_pending2 = pending_groups - 1;
                if (desired_pending2 < 0) {
                    desired_pending2 = 0;
                }
                if (desired_pending2 > (STAGES - 1)) {
                    desired_pending2 = (STAGES - 1);
                }
                op::paged_attention::cuda::cpAsyncWaitGroupRt(desired_pending2);
                pending_groups = desired_pending2;
                __syncthreads();
            }
        }

        op::paged_attention::cuda::cpAsyncWaitAll();
        __syncthreads();
    }

    if (is_active) {
        const size_t idx = static_cast<size_t>(split_idx) * n + base;
        if (lane == 0) {
            partial_m[idx] = m;
            partial_l[idx] = l;
        }
#pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; ++i) {
            const int dim = lane * DIMS_PER_THREAD + i;
            partial_acc[idx * HEAD_SIZE + dim] = acc[i];
        }
    }
}

template <typename Tdata, int HEAD_SIZE>
__device__ void PagedAttentionPrefillSplitKvCombineWarpKernel(
    Tdata *out_,
    const float *partial_acc, // [num_splits, total_q_tokens, num_heads, head_size]
    const float *partial_m,   // [num_splits, total_q_tokens, num_heads]
    const float *partial_l,   // [num_splits, total_q_tokens, num_heads]
    int num_splits,
    size_t total_q_tokens,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {

    const int head_idx = static_cast<int>(blockIdx.x);
    const int token_idx = static_cast<int>(blockIdx.y);
    const int lane = threadIdx.x;
    constexpr int kWarpSize = 32;
    static_assert(HEAD_SIZE % kWarpSize == 0, "HEAD_SIZE must be divisible by 32.");
    constexpr int DIMS_PER_THREAD = HEAD_SIZE / kWarpSize;

    const int num_heads = gridDim.x;
    const size_t n = total_q_tokens * static_cast<size_t>(num_heads);
    const size_t base = static_cast<size_t>(token_idx) * static_cast<size_t>(num_heads) + static_cast<size_t>(head_idx);

    float m = -INFINITY;
    if (lane == 0) {
        for (int s = 0; s < num_splits; ++s) {
            m = fmaxf(m, partial_m[static_cast<size_t>(s) * n + base]);
        }
    }
    m = __shfl_sync(0xffffffff, m, 0);

    float l = 0.0f;
    if (lane == 0) {
        for (int s = 0; s < num_splits; ++s) {
            const float ms = partial_m[static_cast<size_t>(s) * n + base];
            const float ls = partial_l[static_cast<size_t>(s) * n + base];
            if (ls > 0.0f) {
                l += ls * exp2f(ms - m);
            }
        }
    }
    l = __shfl_sync(0xffffffff, l, 0);
    const float inv_l = 1.0f / (l + 1e-6f);

    Tdata *out_ptr = out_ + static_cast<int64_t>(token_idx) * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;
#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        float acc = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            const float ms = partial_m[static_cast<size_t>(s) * n + base];
            const float w = exp2f(ms - m);
            acc += partial_acc[(static_cast<size_t>(s) * n + base) * HEAD_SIZE + dim] * w;
        }
        const float o = acc * inv_l;
        if constexpr (std::is_same_v<Tdata, half>) {
            out_ptr[dim] = __float2half_rn(o);
        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
            out_ptr[dim] = __float2bfloat16_rn(o);
        } else {
            out_ptr[dim] = static_cast<Tdata>(o);
        }
    }
}

// Variant for large K tile where (K+V) shared memory would exceed the per-block limit on some GPUs.
// We keep K in shared memory for reuse across warps, but load V directly from global memory.
template <typename Tindex, typename Tdata, int HEAD_SIZE, int BLOCK_M, int BLOCK_N>
__device__ void PagedAttentionPrefillWarpCtaKernelKOnly(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cu_seqlens_q_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {

    static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128, "Only head_size 64/128 supported in v0.4.");
    static_assert(BLOCK_M > 0 && BLOCK_M <= 16, "BLOCK_M must be <=16.");
    static_assert(BLOCK_N > 0 && BLOCK_N <= 128, "BLOCK_N must be <=128.");

    constexpr int kWarpSize = 32;
    constexpr int DIMS_PER_THREAD = HEAD_SIZE / kWarpSize;
    static_assert(HEAD_SIZE % kWarpSize == 0, "HEAD_SIZE must be divisible by 32.");

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    if (warp_id >= BLOCK_M) {
        return;
    }

    const int head_idx = static_cast<int>(blockIdx.x);
    const int seq_idx = static_cast<int>(blockIdx.y);
    const int m_block = static_cast<int>(blockIdx.z);

    const Tindex q_start = cu_seqlens_q_[seq_idx];
    const Tindex q_end = cu_seqlens_q_[seq_idx + 1];
    const int q_len = static_cast<int>(q_end - q_start);
    if (q_len <= 0) {
        return;
    }

    const int m_start = m_block * BLOCK_M;
    const int q_token_local = m_start + warp_id;
    // IMPORTANT: do not early-return for a subset of warps in this CTA because we use __syncthreads()
    // later. Tail tiles are handled by masking inactive warps.
    if (m_start >= q_len) {
        return; // uniform across the CTA
    }
    const bool is_active = (q_token_local < q_len);

    const int kv_len_total = static_cast<int>(total_kv_lens_[seq_idx]);
    const int history_len = kv_len_total - q_len;
    const int allowed_k_len = is_active ? (history_len + q_token_local + 1) : 0;

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    int64_t q_token = q_start;
    if (is_active) {
        q_token += static_cast<int64_t>(q_token_local);
    }

    const Tindex *block_table = block_tables_ + static_cast<int64_t>(seq_idx) * static_cast<int64_t>(block_table_batch_stride);

    const Tdata *q_ptr = nullptr;
    Tdata *out_ptr = nullptr;
    if (is_active) {
        q_ptr = q_ + q_token * q_stride + static_cast<int64_t>(head_idx) * q_head_stride;
        out_ptr = out_ + q_token * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;
    }

    float q_reg[DIMS_PER_THREAD];
    float acc[DIMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        q_reg[i] = is_active ? static_cast<float>(q_ptr[dim]) : 0.0f;
        acc[i] = 0.0f;
    }

#if defined(__CUDA_ARCH__)
    float2 q_reg2[DIMS_PER_THREAD / 2];
#pragma unroll
    for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
        q_reg2[j] = make_float2(q_reg[j * 2 + 0], q_reg[j * 2 + 1]);
    }
#endif

    float m = -INFINITY;
    float l = 0.0f;

    const int max_q_in_tile = min(m_start + BLOCK_M, q_len);
    const int max_allowed_k_len = min(history_len + max_q_in_tile, kv_len_total);

    __shared__ int32_t s_phys[BLOCK_N];
    __shared__ int32_t s_off[BLOCK_N];
    __shared__ __align__(16) Tdata s_k[BLOCK_N * HEAD_SIZE];

    const int pbs = static_cast<int>(page_block_size);

    for (int k_base = 0; k_base < max_allowed_k_len; k_base += BLOCK_N) {
        const int tile_n = min(BLOCK_N, max_allowed_k_len - k_base);

        for (int t = threadIdx.x; t < tile_n; t += blockDim.x) {
            const int kpos = k_base + t;
            const int page = (pbs == 256) ? (kpos >> 8) : (kpos / pbs);
            const int off = (pbs == 256) ? (kpos & 255) : (kpos - page * pbs);
            const int32_t phys = static_cast<int32_t>(block_table[page]);
            s_phys[t] = phys;
            s_off[t] = off;
        }
        __syncthreads();

        const int tile_elems = tile_n * HEAD_SIZE;
        for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            const int t = idx / HEAD_SIZE;
            const int dim = idx - t * HEAD_SIZE;
            const int32_t phys = s_phys[t];
            const int32_t off = s_off[t];
            const Tdata *k_base_ptr = k_cache_ + static_cast<int64_t>(phys) * k_batch_stride + static_cast<int64_t>(off) * k_row_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;
            s_k[t * HEAD_SIZE + dim] = k_base_ptr[dim];
        }
        __syncthreads();

        for (int t = 0; t < tile_n; ++t) {
            const int kpos = k_base + t;
            if (kpos >= allowed_k_len) {
                break;
            }
            const Tdata *k_ptr = s_k + t * HEAD_SIZE;
            const int32_t phys = s_phys[t];
            const int32_t off = s_off[t];
            const Tdata *v_ptr = v_cache_ + static_cast<int64_t>(phys) * v_batch_stride + static_cast<int64_t>(off) * v_row_stride + static_cast<int64_t>(kv_head_idx) * v_head_stride;

            float qk = 0.0f;
#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<Tdata, half>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const half2 *k2 = reinterpret_cast<const half2 *>(k_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 qf = q_reg2[j];
                    const float2 kf = __half22float2(k2[j]);
                    qk += qf.x * kf.x + qf.y * kf.y;
                }
            } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const __nv_bfloat162 *k2 = reinterpret_cast<const __nv_bfloat162 *>(k_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 qf = q_reg2[j];
                    const float2 kf = __bfloat1622float2(k2[j]);
                    qk += qf.x * kf.x + qf.y * kf.y;
                }
            } else
#endif
            {
#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    const int dim = lane * DIMS_PER_THREAD + i;
                    qk += q_reg[i] * static_cast<float>(k_ptr[dim]);
                }
            }

            qk = op::paged_attention::cuda::warpReduceSum(qk);

            float alpha = 1.0f;
            float beta = 0.0f;
            if (lane == 0) {
                float score = qk * scale_log2;
                if (alibi_slope != 0.0f) {
                    score += (alibi_slope * static_cast<float>(kpos - (allowed_k_len - 1))) * kLog2e;
                }
                const float m_new = fmaxf(m, score);
                alpha = exp2f(m - m_new);
                beta = exp2f(score - m_new);
                l = l * alpha + beta;
                m = m_new;
            }
            alpha = op::paged_attention::cuda::warpBroadcast(alpha, 0);
            beta = op::paged_attention::cuda::warpBroadcast(beta, 0);

#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<Tdata, half>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const half2 *v2 = reinterpret_cast<const half2 *>(v_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 vf = __half22float2(v2[j]);
                    acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
                    acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
                }
            } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                const int dim_base = lane * DIMS_PER_THREAD;
                const __nv_bfloat162 *v2 = reinterpret_cast<const __nv_bfloat162 *>(v_ptr + dim_base);
#pragma unroll
                for (int j = 0; j < DIMS_PER_THREAD / 2; ++j) {
                    const float2 vf = __bfloat1622float2(v2[j]);
                    acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
                    acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
                }
            } else
#endif
            {
#pragma unroll
                for (int i = 0; i < DIMS_PER_THREAD; ++i) {
                    const int dim = lane * DIMS_PER_THREAD + i;
                    const float v_val = static_cast<float>(v_ptr[dim]);
                    acc[i] = acc[i] * alpha + beta * v_val;
                }
            }
        }

        __syncthreads();
    }

    float inv_l = 0.0f;
    if (lane == 0) {
        inv_l = 1.0f / (l + 1e-6f);
    }
    inv_l = op::paged_attention::cuda::warpBroadcast(inv_l, 0);

#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        const float out_val = acc[i] * inv_l;
        if (!is_active) {
            continue;
        }
        if constexpr (std::is_same_v<Tdata, half>) {
            out_ptr[dim] = __float2half_rn(out_val);
        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
            out_ptr[dim] = __float2bfloat16_rn(out_val);
        } else {
            out_ptr[dim] = static_cast<Tdata>(out_val);
        }
    }
}

// TensorCore (WMMA) score kernel (v0.4 experimental):
// - Target shape: head_dim=128, page_block_size=256, fp16.
// - Compute QK^T with WMMA into shared memory, then reuse the existing online-softmax + V accumulation
//   pattern (SIMT) per query row.
//
// Notes:
// - This is a correctness-first kernel. It doesn't yet use MMA for PV (P * V) update.
// - We keep the same grid mapping as other prefill kernels: blockIdx = (head, seq, m_block).
template <int kWarpSize, int kBlockN, int kHeadDim, int kDimsPerThread>
__device__ __forceinline__ void PagedAttentionPrefillMmaScoreUpdateRow(
    int lane,
    int k_base,
    int allowed_k_len,
    const float *scores_row, // [kBlockN]
    const half *v_tile,      // [kBlockN, kHeadDim]
    float scale_log2,
    float alibi_slope_log2,
    float &m,
    float &l,
    float *acc) { // [kDimsPerThread]

    // Max over keys in this tile.
    float local_max = -INFINITY;
    for (int t = lane; t < kBlockN; t += kWarpSize) {
        const int kpos = k_base + t;
        if (kpos >= allowed_k_len) {
            continue;
        }
        float score = scores_row[t] * scale_log2;
        if (alibi_slope_log2 != 0.0f) {
            score += alibi_slope_log2 * static_cast<float>(kpos - (allowed_k_len - 1));
        }
        local_max = fmaxf(local_max, score);
    }
    float tile_m = op::paged_attention::cuda::warpReduceMax(local_max);
    tile_m = __shfl_sync(0xffffffff, tile_m, 0);

    // Sumexp + weighted V over keys in this tile, partitioned by lanes.
    float sumexp_lane = 0.0f;
    float acc_tile[kDimsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int dim_base = lane * kDimsPerThread;
    if (tile_m != -INFINITY) {
        for (int t = lane; t < kBlockN; t += kWarpSize) {
            const int kpos = k_base + t;
            if (kpos >= allowed_k_len) {
                continue;
            }
            float score = scores_row[t] * scale_log2;
            if (alibi_slope_log2 != 0.0f) {
                score += alibi_slope_log2 * static_cast<float>(kpos - (allowed_k_len - 1));
            }
            const float w = exp2f(score - tile_m);
            sumexp_lane += w;

            const half *v_ptr = v_tile + t * kHeadDim + dim_base;
            const half2 *v2 = reinterpret_cast<const half2 *>(v_ptr);
#pragma unroll
            for (int j = 0; j < kDimsPerThread / 2; ++j) {
                const float2 vf = __half22float2(v2[j]);
                acc_tile[j * 2 + 0] += w * vf.x;
                acc_tile[j * 2 + 1] += w * vf.y;
            }
        }
    }

    float tile_sumexp = op::paged_attention::cuda::warpReduceSum(sumexp_lane);
    tile_sumexp = __shfl_sync(0xffffffff, tile_sumexp, 0);

    float alpha = 1.0f;
    float beta = 0.0f;
    if (lane == 0) {
        if (tile_sumexp > 0.0f && tile_m != -INFINITY) {
            const float m_new = fmaxf(m, tile_m);
            alpha = exp2f(m - m_new);
            beta = exp2f(tile_m - m_new);
            l = l * alpha + tile_sumexp * beta;
            m = m_new;
        } else {
            alpha = 1.0f;
            beta = 0.0f;
        }
    }
    alpha = __shfl_sync(0xffffffff, alpha, 0);
    beta = __shfl_sync(0xffffffff, beta, 0);
#pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
        acc[i] = acc[i] * alpha + beta * acc_tile[i];
    }
}

template <typename Tindex, int kWarpSize, int kHeadDim, int kDimsPerThread>
__device__ __forceinline__ void PagedAttentionPrefillMmaScoreWriteRow(
    int lane,
    bool active,
    int q_token_local,
    Tindex q_start,
    int head_idx,
    half *out_,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    float l,
    const float *acc) { // [kDimsPerThread]
    if (!active) {
        return;
    }

    float inv_l = 0.0f;
    if (lane == 0) {
        inv_l = 1.0f / (l + 1e-6f);
    }
    inv_l = op::paged_attention::cuda::warpBroadcast(inv_l, 0);

    const int64_t q_token = q_start + static_cast<int64_t>(q_token_local);
    half *out_ptr = out_ + q_token * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;
#pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
        const int dim = lane * kDimsPerThread + i;
        out_ptr[dim] = __float2half_rn(acc[i] * inv_l);
    }
}

template <typename Tindex>
__device__ void PagedAttentionPrefillWarpCta8MmaHd128Kernel(
    half *out_,
    const half *q_,
    const half *k_cache_,
    const half *v_cache_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const Tindex *cu_seqlens_q_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {

    (void)max_num_blocks_per_seq;

    constexpr int kWarpSize = 32;
    constexpr int kWarps = 8;
    constexpr int kHeadDim = 128;
    // Extra padding in the K dimension to reduce shared-memory bank conflicts for ldmatrix / wmma loads.
    // NOTE: FA2 uses a swizzled smem layout; padding is a smaller step that keeps our code simple.
    constexpr int kHeadDimSmem = 136; // must be a multiple of 8 for wmma::load_matrix_sync
    constexpr int kBlockM = 16;       // 2 rows per warp
    // Keep static shared memory <= 48KB for compatibility with build targets that cap SMEM at 0xC000.
    // kBlockN=64 brings s_q+s_k+s_v+s_scores+s_phys/s_off down to ~41KB.
    constexpr int kBlockN = 64;
    constexpr int kDimsPerThread = kHeadDim / kWarpSize;

    static_assert(kHeadDim % kWarpSize == 0, "head_dim must be divisible by 32.");

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    if (warp_id >= kWarps) {
        return;
    }

    const int head_idx = static_cast<int>(blockIdx.x);
    const int seq_idx = static_cast<int>(blockIdx.y);
    const int m_block = static_cast<int>(blockIdx.z);

    const Tindex q_start = cu_seqlens_q_[seq_idx];
    const Tindex q_end = cu_seqlens_q_[seq_idx + 1];
    const int q_len = static_cast<int>(q_end - q_start);
    if (q_len <= 0) {
        return;
    }

    const int m_start = m_block * kBlockM;
    // Uniform early return for empty tail tiles (avoid deadlock with __syncthreads()).
    if (m_start >= q_len) {
        return;
    }

    const int kv_len_total = static_cast<int>(total_kv_lens_[seq_idx]);
    const int history_len = kv_len_total - q_len;

    // Clamp max k length for this CTA based on the last active query row in the tile.
    const int max_q_in_tile = min(m_start + kBlockM, q_len);
    const int max_allowed_k_len = min(history_len + max_q_in_tile, kv_len_total);

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;
    const float alibi_slope_log2 = alibi_slope * kLog2e;

    const int pbs = static_cast<int>(page_block_size);

    const Tindex *block_table = block_tables_ + static_cast<int64_t>(seq_idx) * static_cast<int64_t>(block_table_batch_stride);

    // Shared memory:
    // - s_q: [kBlockM, kHeadDimSmem] (padded)
    // - s_k/s_v: [kBlockN, kHeadDim]
    // - s_scores: [kBlockM, kBlockN] raw dot products (no scale / alibi)
    __shared__ __align__(16) half s_q[kBlockM * kHeadDimSmem];
    __shared__ int32_t s_phys[kBlockN];
    __shared__ int32_t s_off[kBlockN];
    __shared__ __align__(16) half s_k[kBlockN * kHeadDimSmem];
    __shared__ __align__(16) half s_v[kBlockN * kHeadDimSmem];
    __shared__ __align__(16) float s_scores[kBlockM * kBlockN];

    // Load Q tile (pad inactive rows with 0).
    for (int idx = threadIdx.x; idx < kBlockM * kHeadDim; idx += blockDim.x) {
        const int r = idx / kHeadDim;
        const int d = idx - r * kHeadDim;
        const int q_token_local = m_start + r;
        if (q_token_local < q_len) {
            const int64_t q_token = q_start + static_cast<int64_t>(q_token_local);
            const half *q_ptr = q_ + q_token * q_stride + static_cast<int64_t>(head_idx) * q_head_stride;
            s_q[r * kHeadDimSmem + d] = q_ptr[d];
        } else {
            s_q[r * kHeadDimSmem + d] = __float2half_rn(0.0f);
        }
    }
    __syncthreads();

    // Two rows per warp: row0=warp_id, row1=warp_id+kWarps.
    const int row0 = warp_id;
    const int row1 = warp_id + kWarps;
    const bool active0 = (row0 < kBlockM) && ((m_start + row0) < q_len);
    const bool active1 = (row1 < kBlockM) && ((m_start + row1) < q_len);
    const int allowed0 = active0 ? min(history_len + (m_start + row0) + 1, kv_len_total) : 0;
    const int allowed1 = active1 ? min(history_len + (m_start + row1) + 1, kv_len_total) : 0;

    float m0 = -INFINITY, l0 = 0.0f;
    float m1 = -INFINITY, l1 = 0.0f;
    float acc0[kDimsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[kDimsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Iterate over K/V tiles.
    for (int k_base = 0; k_base < max_allowed_k_len; k_base += kBlockN) {
        // Map logical k positions to physical blocks for this tile (pad the tail with -1).
        for (int t = threadIdx.x; t < kBlockN; t += blockDim.x) {
            const int kpos = k_base + t;
            if (kpos < max_allowed_k_len) {
                const int page = (pbs == 256) ? (kpos >> 8) : (kpos / pbs);
                const int off = (pbs == 256) ? (kpos & 255) : (kpos - page * pbs);
                s_phys[t] = static_cast<int32_t>(block_table[page]);
                s_off[t] = off;
            } else {
                s_phys[t] = -1;
                s_off[t] = 0;
            }
        }
        __syncthreads();

        // Load K/V tile into shared memory (pad with 0 for inactive tokens).
        for (int idx = threadIdx.x; idx < kBlockN * kHeadDim; idx += blockDim.x) {
            const int t = idx / kHeadDim;
            const int d = idx - t * kHeadDim;
            const int32_t phys = s_phys[t];
            if (phys >= 0) {
                const int32_t off = s_off[t];
                const half *k_ptr = k_cache_ + static_cast<int64_t>(phys) * k_batch_stride + static_cast<int64_t>(off) * k_row_stride + static_cast<int64_t>(kv_head_idx) * k_head_stride;
                const half *v_ptr = v_cache_ + static_cast<int64_t>(phys) * v_batch_stride + static_cast<int64_t>(off) * v_row_stride + static_cast<int64_t>(kv_head_idx) * v_head_stride;
                s_k[t * kHeadDimSmem + d] = k_ptr[d];
                s_v[t * kHeadDimSmem + d] = v_ptr[d];
            } else {
                s_k[t * kHeadDimSmem + d] = __float2half_rn(0.0f);
                s_v[t * kHeadDimSmem + d] = __float2half_rn(0.0f);
            }
        }
        __syncthreads();

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && !defined(ENABLE_HYGON_API)
        // WMMA: each warp computes scores for 16 keys (one 16-column slice of the K tile) across all 16 rows.
        // For kBlockN=64, only the first 4 warps participate in WMMA score computation.
        // Hygon DCU does not provide nvcuda::wmma; gate it out and fall through to the CUDA-core path.
        namespace wmma = nvcuda::wmma;
        constexpr int kNSub = kBlockN / 16;
        if (warp_id < kNSub) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            const int n_sub = warp_id; // [0, kNSub)
            const half *q_tile = s_q;
            const half *k_tile = s_k + (n_sub * 16) * kHeadDimSmem;
            // K loop (head_dim=128).
#pragma unroll
            for (int kk = 0; kk < (kHeadDim / 16); ++kk) {
                wmma::load_matrix_sync(a_frag, q_tile + kk * 16, kHeadDimSmem);
                wmma::load_matrix_sync(b_frag, k_tile + kk * 16, kHeadDimSmem);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            float *scores_tile = s_scores + n_sub * 16;
            wmma::store_matrix_sync(scores_tile, c_frag, kBlockN, wmma::mem_row_major);
        }
#else
        // No WMMA support on this architecture: fall back to scalar dot in the existing kernels.
        // (We keep scores as 0 so this kernel is effectively incorrect; host dispatch must avoid selecting it.)
        if (threadIdx.x == 0) {
            // Intentionally empty.
        }
#endif
        __syncthreads();

        // Online softmax + V update per row handled by the same warp across tiles.
        if (row0 < kBlockM) {
            PagedAttentionPrefillMmaScoreUpdateRow<kWarpSize, kBlockN, kHeadDim, kDimsPerThread>(
                lane, k_base, allowed0, s_scores + row0 * kBlockN, s_v, scale_log2, alibi_slope_log2, m0, l0, acc0);
        }
        if (row1 < kBlockM) {
            PagedAttentionPrefillMmaScoreUpdateRow<kWarpSize, kBlockN, kHeadDim, kDimsPerThread>(
                lane, k_base, allowed1, s_scores + row1 * kBlockN, s_v, scale_log2, alibi_slope_log2, m1, l1, acc1);
        }
        __syncthreads();
    }

    // Write outputs.
    if (row0 < kBlockM) {
        PagedAttentionPrefillMmaScoreWriteRow<Tindex, kWarpSize, kHeadDim, kDimsPerThread>(
            lane, active0, m_start + row0, q_start, head_idx, out_, o_stride, o_head_stride, l0, acc0);
    }
    if (row1 < kBlockM) {
        PagedAttentionPrefillMmaScoreWriteRow<Tindex, kWarpSize, kHeadDim, kDimsPerThread>(
            lane, active1, m_start + row1, q_start, head_idx, out_, o_stride, o_head_stride, l1, acc1);
    }
}

} // namespace op::paged_attention_prefill::cuda

#endif
