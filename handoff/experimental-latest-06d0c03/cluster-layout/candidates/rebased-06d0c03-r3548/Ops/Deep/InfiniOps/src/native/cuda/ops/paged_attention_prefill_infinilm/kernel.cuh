#ifndef INFINI_OPS_CUDA_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_CUH_
#define INFINI_OPS_CUDA_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_CUH_

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ALI_API) || \
    defined(WITH_ILUVATAR) || defined(ENABLE_ILUVATAR_API)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#endif

#include <cstdint>
#include <type_traits>

#include "native/cuda/caster.cuh"

// Reuse warp-level primitives and math helpers from decode flash_attention
// kernels.
#include "native/cuda/ops/paged_attention_infinilm/kernel.cuh"

namespace op::paged_attention_prefill::cuda {

template <typename TIndex>
struct CumulativeSequenceLengths {
  const TIndex* values;

  __device__ __forceinline__ TIndex operator[](size_t index) const {
    return values[index + 1] - values[index];
  }
};

template <typename TIndex>
__device__ __forceinline__ size_t FindSeqId(size_t token_idx,
                                            const TIndex* cu_seqlens_q,
                                            size_t num_seqs) {
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

template <infini::ops::Device::Type kDev, typename TIndex, typename TData,
          int kHeadSize>
__device__ void PagedAttentionPrefillWarpKernel(
    TData* out, const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, const TIndex* total_kv_lens,
    const TIndex* cu_seqlens_q, const float* alibi_slopes, size_t num_kv_heads,
    float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t block_table_batch_stride, ptrdiff_t qstride,
    ptrdiff_t qhead_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride, ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride, ptrdiff_t o_stride, ptrdiff_t o_head_stride) {
  constexpr int kWarpSize = 32;
  static_assert(kHeadSize == 64 || kHeadSize == 128 || kHeadSize == 192,
                "Only head_size 64/128/192 supported in v0.4.");
  static_assert(kHeadSize % kWarpSize == 0,
                "kHeadSize must be divisible by 32.");
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;

  const int lane = threadIdx.x;

  const int head_idx = static_cast<int>(blockIdx.x);
  const int seqidx = static_cast<int>(blockIdx.y);
  const int qtoken_local = static_cast<int>(blockIdx.z);

  const TIndex qstart = cu_seqlens_q[seqidx];
  const TIndex qend = cu_seqlens_q[seqidx + 1];
  const int qlen = static_cast<int>(qend - qstart);
  if (qtoken_local >= qlen) {
    return;
  }

  const int kv_len_total = static_cast<int>(total_kv_lens[seqidx]);
  const int history_len = kv_len_total - qlen;
  const int allowed_k_len = history_len + qtoken_local + 1;
  if (allowed_k_len <= 0) {
    return;
  }

  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  const int64_t qtoken = qstart + static_cast<int64_t>(qtoken_local);
  const TData* qptr =
      q + qtoken * qstride + static_cast<int64_t>(head_idx) * qhead_stride;
  TData* outptr =
      out + qtoken * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;

  const TIndex* block_table =
      block_tables + static_cast<int64_t>(seqidx) *
                         static_cast<int64_t>(block_table_batch_stride);

  float qreg[kDimsPerThread];
  float acc[kDimsPerThread];
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    qreg[i] = static_cast<float>(qptr[dim]);
    acc[i] = 0.0f;
  }

#if defined(__CUDA_ARCH__)
  float2 qreg2[kDimsPerThread / 2];
  if constexpr (std::is_same_v<TData, half>) {
    const int dim_base = lane * kDimsPerThread;
    const half2* q2 = reinterpret_cast<const half2*>(qptr + dim_base);
#pragma unroll
    for (int j = 0; j < kDimsPerThread / 2; ++j) {
      qreg2[j] = __half22float2(q2[j]);
    }
  }
  if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
    const int dim_base = lane * kDimsPerThread;
    const __nv_bfloat162* q2 =
        reinterpret_cast<const __nv_bfloat162*>(qptr + dim_base);
#pragma unroll
    for (int j = 0; j < kDimsPerThread / 2; ++j) {
      qreg2[j] = __bfloat1622float2(q2[j]);
    }
  }
#endif

  float m = -INFINITY;
  float l = 0.0f;

  const int pbs = static_cast<int>(page_block_size);
  int t_base = 0;
  for (int logical_block = 0; t_base < allowed_k_len;
       ++logical_block, t_base += pbs) {
    int physical_block = 0;
    if (lane == 0) {
      physical_block = static_cast<int>(block_table[logical_block]);
    }
    physical_block = __shfl_sync(0xffffffff, physical_block, 0);

    const TData* k_base =
        k_cache + static_cast<int64_t>(physical_block) * k_batch_stride +
        static_cast<int64_t>(kv_head_idx) * k_head_stride;
    const TData* v_base =
        v_cache + static_cast<int64_t>(physical_block) * v_batch_stride +
        static_cast<int64_t>(kv_head_idx) * v_head_stride;

    const int token_end = min(pbs, allowed_k_len - t_base);
    for (int token_in_block = 0; token_in_block < token_end; ++token_in_block) {
      const int t = t_base + token_in_block;
      const TData* k_ptr =
          k_base + static_cast<int64_t>(token_in_block) * k_row_stride;
      const TData* v_ptr =
          v_base + static_cast<int64_t>(token_in_block) * v_row_stride;

      float qk = 0.0f;
#if defined(__CUDA_ARCH__)
      if constexpr (std::is_same_v<TData, half>) {
        const int dim_base = lane * kDimsPerThread;
        const half2* k2 = reinterpret_cast<const half2*>(k_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 qf = qreg2[j];
          const float2 kf = __half22float2(k2[j]);
          qk += qf.x * kf.x + qf.y * kf.y;
        }
      } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
        const int dim_base = lane * kDimsPerThread;
        const __nv_bfloat162* k2 =
            reinterpret_cast<const __nv_bfloat162*>(k_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 qf = qreg2[j];
          const float2 kf = __bfloat1622float2(k2[j]);
          qk += qf.x * kf.x + qf.y * kf.y;
        }
      } else
#endif
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
          const int dim = lane * kDimsPerThread + i;
          qk += qreg[i] * static_cast<float>(k_ptr[dim]);
        }
      qk = op::paged_attention::cuda::WarpReduceSum(qk);

      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0) {
        float score = qk * scale_log2;
        if (alibi_slope != 0.0f) {
          const int causal_limit = allowed_k_len - 1;
          score +=
              (alibi_slope * static_cast<float>(t - causal_limit)) * kLog2e;
        }
        const float m_new = fmaxf(m, score);
        alpha = exp2f(m - m_new);
        beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;
      }
      alpha = op::paged_attention::cuda::WarpBroadcast(alpha, 0);
      beta = op::paged_attention::cuda::WarpBroadcast(beta, 0);

#if defined(__CUDA_ARCH__)
      if constexpr (std::is_same_v<TData, half>) {
        const int dim_base = lane * kDimsPerThread;
        const half2* v2 = reinterpret_cast<const half2*>(v_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 vf = __half22float2(v2[j]);
          acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
          acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
        }
      } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
        const int dim_base = lane * kDimsPerThread;
        const __nv_bfloat162* v2 =
            reinterpret_cast<const __nv_bfloat162*>(v_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 vf = __bfloat1622float2(v2[j]);
          acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
          acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
        }
      } else
#endif
      {
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
          const int dim = lane * kDimsPerThread + i;
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
  inv_l = op::paged_attention::cuda::WarpBroadcast(inv_l, 0);

#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    const float o = acc[i] * inv_l;
    outptr[dim] = infini::ops::Caster<kDev>::template Cast<TData>(o);
  }
}

template <infini::ops::Device::Type kDev, typename TIndex, typename TData,
          int kHeadSize, typename TLengths>
__global__ void PagedAttentionPrefillWarpGlobalKernel(
    TData* out, const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, TLengths total_kv_lens,
    const TIndex* cu_seqlens_q, const float* alibi_slopes, size_t num_heads,
    size_t num_seqs, size_t num_kv_heads, size_t total_qtokens, float scale,
    size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t block_table_batch_stride, ptrdiff_t qstride,
    ptrdiff_t qhead_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride, ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride, ptrdiff_t o_stride, ptrdiff_t o_head_stride) {
  constexpr int kWarpSize = 32;
  static_assert(kHeadSize == 64 || kHeadSize == 128 || kHeadSize == 192,
                "Only head_size 64/128/192 supported in v0.4.");
  static_assert(kHeadSize % kWarpSize == 0,
                "kHeadSize must be divisible by 32.");
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;

  const int lane = threadIdx.x;
  const size_t head_idx = static_cast<size_t>(blockIdx.x);
  const size_t global_token_idx = static_cast<size_t>(blockIdx.y) +
                                  static_cast<size_t>(blockIdx.z) * gridDim.y;

  if (lane >= kWarpSize || head_idx >= num_heads ||
      global_token_idx >= total_qtokens) {
    return;
  }

  const size_t seqidx =
      FindSeqId<TIndex>(global_token_idx, cu_seqlens_q, num_seqs);
  const TIndex qstart = cu_seqlens_q[seqidx];
  const TIndex qend = cu_seqlens_q[seqidx + 1];
  const int qlen = static_cast<int>(qend - qstart);

  const int qtoken_local =
      static_cast<int>(global_token_idx - static_cast<size_t>(qstart));
  if (qtoken_local < 0 || qtoken_local >= qlen) {
    return;
  }

  const int kv_len_total = static_cast<int>(total_kv_lens[seqidx]);
  const int history_len = kv_len_total - qlen;
  const int allowed_k_len = history_len + qtoken_local + 1;
  TData* outptr = out + static_cast<int64_t>(global_token_idx) * o_stride +
                  static_cast<int64_t>(head_idx) * o_head_stride;
  if (allowed_k_len <= 0) {
#pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
      const int dim = lane * kDimsPerThread + i;
      outptr[dim] = TData{};
    }
    return;
  }

  const int num_queries_per_kv = static_cast<int>(num_heads / num_kv_heads);
  const int kv_head_idx = static_cast<int>(head_idx) / num_queries_per_kv;

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  const TData* qptr = q + static_cast<int64_t>(global_token_idx) * qstride +
                      static_cast<int64_t>(head_idx) * qhead_stride;

  const TIndex* block_table =
      block_tables + static_cast<int64_t>(seqidx) *
                         static_cast<int64_t>(block_table_batch_stride);
  const int pbs = static_cast<int>(page_block_size);

  float qreg[kDimsPerThread];
  float acc[kDimsPerThread];
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    qreg[i] = static_cast<float>(qptr[dim]);
    acc[i] = 0.0f;
  }

#if defined(__CUDA_ARCH__)
  float2 qreg2[kDimsPerThread / 2];
  if constexpr (std::is_same_v<TData, half>) {
    const int dim_base = lane * kDimsPerThread;
    const half2* q2 = reinterpret_cast<const half2*>(qptr + dim_base);
#pragma unroll
    for (int j = 0; j < kDimsPerThread / 2; ++j) {
      qreg2[j] = __half22float2(q2[j]);
    }
  }
  if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
    const int dim_base = lane * kDimsPerThread;
    const __nv_bfloat162* q2 =
        reinterpret_cast<const __nv_bfloat162*>(qptr + dim_base);
#pragma unroll
    for (int j = 0; j < kDimsPerThread / 2; ++j) {
      qreg2[j] = __bfloat1622float2(q2[j]);
    }
  }
#endif

  float m = -INFINITY;
  float l = 0.0f;

  // Iterate by pages to avoid per-token division/mod and redundant block_table
  // loads.
  int t_base = 0;
  for (int logical_block = 0;
       t_base < allowed_k_len &&
       logical_block < static_cast<int>(max_num_blocks_per_seq);
       ++logical_block, t_base += pbs) {
    const int32_t phys = static_cast<int32_t>(block_table[logical_block]);
    const TData* k_base = k_cache +
                          static_cast<int64_t>(phys) * k_batch_stride +
                          static_cast<int64_t>(kv_head_idx) * k_head_stride;
    const TData* v_base = v_cache +
                          static_cast<int64_t>(phys) * v_batch_stride +
                          static_cast<int64_t>(kv_head_idx) * v_head_stride;

    const int token_end = min(pbs, allowed_k_len - t_base);
    for (int token_in_block = 0; token_in_block < token_end; ++token_in_block) {
      const int t = t_base + token_in_block;
      const TData* k_ptr =
          k_base + static_cast<int64_t>(token_in_block) * k_row_stride;
      const TData* v_ptr =
          v_base + static_cast<int64_t>(token_in_block) * v_row_stride;

      float qk = 0.0f;
#if defined(__CUDA_ARCH__)
      if constexpr (std::is_same_v<TData, half>) {
        const int dim_base = lane * kDimsPerThread;
        const half2* k2 = reinterpret_cast<const half2*>(k_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 qf = qreg2[j];
          const float2 kf = __half22float2(k2[j]);
          qk += qf.x * kf.x + qf.y * kf.y;
        }
      } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
        const int dim_base = lane * kDimsPerThread;
        const __nv_bfloat162* k2 =
            reinterpret_cast<const __nv_bfloat162*>(k_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 qf = qreg2[j];
          const float2 kf = __bfloat1622float2(k2[j]);
          qk += qf.x * kf.x + qf.y * kf.y;
        }
      } else
#endif
      {
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
          const int dim = lane * kDimsPerThread + i;
          qk += qreg[i] * static_cast<float>(k_ptr[dim]);
        }
      }
      qk = op::paged_attention::cuda::WarpReduceSum(qk);

      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0) {
        float score = qk * scale_log2;
        if (alibi_slope != 0.0f) {
          const int causal_limit = allowed_k_len - 1;
          score +=
              (alibi_slope * static_cast<float>(t - causal_limit)) * kLog2e;
        }
        const float m_new = fmaxf(m, score);
        alpha = exp2f(m - m_new);
        beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;
      }
      alpha = op::paged_attention::cuda::WarpBroadcast(alpha, 0);
      beta = op::paged_attention::cuda::WarpBroadcast(beta, 0);

#if defined(__CUDA_ARCH__)
      if constexpr (std::is_same_v<TData, half>) {
        const int dim_base = lane * kDimsPerThread;
        const half2* v2 = reinterpret_cast<const half2*>(v_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 vf = __half22float2(v2[j]);
          acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
          acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
        }
      } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
        const int dim_base = lane * kDimsPerThread;
        const __nv_bfloat162* v2 =
            reinterpret_cast<const __nv_bfloat162*>(v_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 vf = __bfloat1622float2(v2[j]);
          acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
          acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
        }
      } else
#endif
      {
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
          const int dim = lane * kDimsPerThread + i;
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
#ifdef INFINI_OPS_PAGED_ATTENTION_ILUVATAR
  inv_l = op::paged_attention::cuda::WarpBroadcast(inv_l, 0);
#else
  inv_l = __shfl_sync(0xffffffff, inv_l, 0);
#endif

#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    const float o = acc[i] * inv_l;
    outptr[dim] = infini::ops::Caster<kDev>::template Cast<TData>(o);
  }
}

template <typename TIndex, typename TData, typename TCompute, int kHeadSize>
__global__ void PagedAttentionPrefillReferenceKernel(
    TData* out, const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, const TIndex* total_kv_lens,
    const TIndex* cu_seqlens_q, const float* alibi_slopes, size_t num_heads,
    size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq,
    size_t page_block_size, ptrdiff_t block_table_batch_stride,
    ptrdiff_t qstride, ptrdiff_t qhead_stride, ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride, ptrdiff_t k_head_stride, ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride, ptrdiff_t v_head_stride, ptrdiff_t o_stride,
    ptrdiff_t o_head_stride, size_t num_seqs) {
  const size_t global_token_idx = static_cast<size_t>(blockIdx.x);
  const size_t head_idx = static_cast<size_t>(blockIdx.y);
  const size_t dim_idx = static_cast<size_t>(threadIdx.x);

  if (dim_idx >= kHeadSize || head_idx >= num_heads) {
    return;
  }

  const size_t seqidx =
      FindSeqId<TIndex>(global_token_idx, cu_seqlens_q, num_seqs);
  const size_t qtoken_idx =
      global_token_idx - static_cast<size_t>(cu_seqlens_q[seqidx]);
  const size_t qlen =
      static_cast<size_t>(cu_seqlens_q[seqidx + 1] - cu_seqlens_q[seqidx]);

  const size_t total_kv_len = static_cast<size_t>(total_kv_lens[seqidx]);
  const size_t history_len = total_kv_len - qlen;
  const size_t causal_limit = history_len + qtoken_idx;

  const size_t num_queries_per_kv = num_heads / num_kv_heads;
  const size_t kv_head_idx = head_idx / num_queries_per_kv;

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];

  const TData* qvec = q + static_cast<int64_t>(global_token_idx) * qstride +
                      static_cast<int64_t>(head_idx) * qhead_stride;
  TData* outptr = out + static_cast<int64_t>(global_token_idx) * o_stride +
                  static_cast<int64_t>(head_idx) * o_head_stride;

  const TIndex* block_table =
      block_tables + static_cast<int64_t>(seqidx) *
                         static_cast<int64_t>(block_table_batch_stride);
  const size_t pbs = page_block_size;

  TCompute max_score = -INFINITY;
  for (size_t t = 0; t <= causal_limit; ++t) {
    const size_t page = t / pbs;
    const size_t off = t - page * pbs;
    const ptrdiff_t phys = static_cast<ptrdiff_t>(block_table[page]);
    const TData* k_vec = k_cache + static_cast<int64_t>(phys) * k_batch_stride +
                         static_cast<int64_t>(off) * k_row_stride +
                         static_cast<int64_t>(kv_head_idx) * k_head_stride;

    TCompute score = 0;
    for (size_t d = 0; d < kHeadSize; ++d) {
      score += static_cast<TCompute>(qvec[d]) * static_cast<TCompute>(k_vec[d]);
    }
    score *= static_cast<TCompute>(scale);
    if (alibi_slope != 0.0f) {
      score += static_cast<TCompute>(alibi_slope *
                                     static_cast<float>(t - causal_limit));
    }
    if (score > max_score) {
      max_score = score;
    }
  }

  TCompute sum_exp = 0;
  for (size_t t = 0; t <= causal_limit; ++t) {
    const size_t page = t / pbs;
    const size_t off = t - page * pbs;
    const ptrdiff_t phys = static_cast<ptrdiff_t>(block_table[page]);
    const TData* k_vec = k_cache + static_cast<int64_t>(phys) * k_batch_stride +
                         static_cast<int64_t>(off) * k_row_stride +
                         static_cast<int64_t>(kv_head_idx) * k_head_stride;

    TCompute score = 0;
    for (size_t d = 0; d < kHeadSize; ++d) {
      score += static_cast<TCompute>(qvec[d]) * static_cast<TCompute>(k_vec[d]);
    }
    score *= static_cast<TCompute>(scale);
    if (alibi_slope != 0.0f) {
      score += static_cast<TCompute>(alibi_slope *
                                     static_cast<float>(t - causal_limit));
    }
    sum_exp +=
        static_cast<TCompute>(expf(static_cast<float>(score - max_score)));
  }

  const TCompute inv_sum =
      static_cast<TCompute>(1.0f) / (sum_exp + static_cast<TCompute>(1e-6f));
  TCompute acc = 0;
  for (size_t t = 0; t <= causal_limit; ++t) {
    const size_t page = t / pbs;
    const size_t off = t - page * pbs;
    const ptrdiff_t phys = static_cast<ptrdiff_t>(block_table[page]);
    const TData* k_vec = k_cache + static_cast<int64_t>(phys) * k_batch_stride +
                         static_cast<int64_t>(off) * k_row_stride +
                         static_cast<int64_t>(kv_head_idx) * k_head_stride;

    TCompute score = 0;
    for (size_t d = 0; d < kHeadSize; ++d) {
      score += static_cast<TCompute>(qvec[d]) * static_cast<TCompute>(k_vec[d]);
    }
    score *= static_cast<TCompute>(scale);
    if (alibi_slope != 0.0f) {
      score += static_cast<TCompute>(alibi_slope *
                                     static_cast<float>(t - causal_limit));
    }
    const TCompute prob =
        static_cast<TCompute>(expf(static_cast<float>(score - max_score))) *
        inv_sum;

    const TData* v_vec = v_cache + static_cast<int64_t>(phys) * v_batch_stride +
                         static_cast<int64_t>(off) * v_row_stride +
                         static_cast<int64_t>(kv_head_idx) * v_head_stride;
    acc += prob * static_cast<TCompute>(v_vec[dim_idx]);
  }

  outptr[dim_idx] = static_cast<TData>(acc);
}

template <infini::ops::Device::Type kDev, typename TIndex, typename TData,
          int kHeadSize, int kBlockM, int kBlockN>
__device__ void PagedAttentionPrefillWarpCtaKernel(
    TData* out, const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, const TIndex* total_kv_lens,
    const TIndex* cu_seqlens_q, const float* alibi_slopes, size_t num_kv_heads,
    float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t block_table_batch_stride, ptrdiff_t qstride,
    ptrdiff_t qhead_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride, ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride, ptrdiff_t o_stride, ptrdiff_t o_head_stride) {
  static_assert(kHeadSize == 64 || kHeadSize == 128 || kHeadSize == 192,
                "Only head_size 64/128/192 supported in v0.4.");
  static_assert(kBlockM > 0 && kBlockM <= 16,
                "kBlockM must be small (warp-per-query design).");
  static_assert(kBlockN == 64 || kBlockN == 128,
                "kBlockN must be 64/128 in v0.4.");

  constexpr int kWarpSize = 32;
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;
  static_assert(kHeadSize % kWarpSize == 0,
                "kHeadSize must be divisible by 32.");

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  if (warp_id >= kBlockM) {
    return;
  }

  const int head_idx = static_cast<int>(blockIdx.x);
  const int seqidx = static_cast<int>(blockIdx.y);
  const int m_block = static_cast<int>(blockIdx.z);

  const TIndex qstart = cu_seqlens_q[seqidx];
  const TIndex qend = cu_seqlens_q[seqidx + 1];
  const int qlen = static_cast<int>(qend - qstart);
  if (qlen <= 0) {
    return;
  }

  const int m_start = m_block * kBlockM;
  const int qtoken_local = m_start + warp_id;
  // IMPORTANT: do not early-return for a subset of warps in this CTA because we
  // use __syncthreads() later. Tail tiles are handled by masking inactive
  // warps.
  if (m_start >= qlen) {
    return;  // uniform across the CTA
  }
  const bool is_active = (qtoken_local < qlen);

  const int64_t kv_len_total_i64 = total_kv_lens[seqidx];
  const int kv_len_total = static_cast<int>(kv_len_total_i64);
  // history_len = total_kv_len - qlen (KV already includes current q tokens).
  const int history_len = kv_len_total - qlen;
  const int allowed_k_len = is_active ? (history_len + qtoken_local + 1) : 0;

  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  int64_t qtoken = qstart;
  if (is_active) {
    qtoken += static_cast<int64_t>(qtoken_local);
  }

  const TIndex* block_table =
      block_tables + static_cast<int64_t>(seqidx) *
                         static_cast<int64_t>(block_table_batch_stride);

  const TData* qptr = nullptr;
  TData* outptr = nullptr;
  if (is_active) {
    qptr = q + qtoken * qstride + static_cast<int64_t>(head_idx) * qhead_stride;
    outptr = out + qtoken * o_stride +
             static_cast<int64_t>(head_idx) * o_head_stride;
  }

  float qreg[kDimsPerThread];
  float acc[kDimsPerThread];
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    qreg[i] = is_active ? static_cast<float>(qptr[dim]) : 0.0f;
    acc[i] = 0.0f;
  }

#if defined(__CUDA_ARCH__)
  float2 qreg2[kDimsPerThread / 2];
#pragma unroll
  for (int j = 0; j < kDimsPerThread / 2; ++j) {
    qreg2[j] = make_float2(qreg[j * 2 + 0], qreg[j * 2 + 1]);
  }
#endif

  float m = -INFINITY;
  float l = 0.0f;

  // For this CTA, we only need to scan up to the max allowed k among active
  // warps.
  const int max_qin_tile = min(m_start + kBlockM, qlen);
  const int max_allowed_k_len = min(history_len + max_qin_tile, kv_len_total);

  __shared__ int32_t s_phys[kBlockN];
  __shared__ int32_t s_off[kBlockN];
  // Ensure shared-memory tiles are aligned for half2/bfloat162 vector loads.
  __shared__ __align__(16) TData s_k[kBlockN * kHeadSize];
  __shared__ __align__(16) TData s_v[kBlockN * kHeadSize];

  const int pbs = static_cast<int>(page_block_size);

  for (int k_base = 0; k_base < max_allowed_k_len; k_base += kBlockN) {
    const int tile_n = min(kBlockN, max_allowed_k_len - k_base);

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
    const int tile_elems = tile_n * kHeadSize;
    for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
      const int t = idx / kHeadSize;
      const int dim = idx - t * kHeadSize;
      const int32_t phys = s_phys[t];
      const int32_t off = s_off[t];
      const TData* k_base_ptr =
          k_cache + static_cast<int64_t>(phys) * k_batch_stride +
          static_cast<int64_t>(off) * k_row_stride +
          static_cast<int64_t>(kv_head_idx) * k_head_stride;
      const TData* v_base_ptr =
          v_cache + static_cast<int64_t>(phys) * v_batch_stride +
          static_cast<int64_t>(off) * v_row_stride +
          static_cast<int64_t>(kv_head_idx) * v_head_stride;
      s_k[t * kHeadSize + dim] = k_base_ptr[dim];
      s_v[t * kHeadSize + dim] = v_base_ptr[dim];
    }
    __syncthreads();

    // Each warp processes one query token and scans the K/V tile.
    for (int t = 0; t < tile_n; ++t) {
      const int kpos = k_base + t;
      if (kpos >= allowed_k_len) {
        break;
      }
      const TData* k_ptr = s_k + t * kHeadSize;
      const TData* v_ptr = s_v + t * kHeadSize;

      float qk = 0.0f;
#if defined(__CUDA_ARCH__)
      if constexpr (std::is_same_v<TData, half>) {
        const int dim_base = lane * kDimsPerThread;
        const half2* k2 = reinterpret_cast<const half2*>(k_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 qf = qreg2[j];
          const float2 kf = __half22float2(k2[j]);
          qk += qf.x * kf.x + qf.y * kf.y;
        }
      } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
        const int dim_base = lane * kDimsPerThread;
        const __nv_bfloat162* k2 =
            reinterpret_cast<const __nv_bfloat162*>(k_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 qf = qreg2[j];
          const float2 kf = __bfloat1622float2(k2[j]);
          qk += qf.x * kf.x + qf.y * kf.y;
        }
      } else
#endif
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
          const int dim = lane * kDimsPerThread + i;
          qk += qreg[i] * static_cast<float>(k_ptr[dim]);
        }

      qk = op::paged_attention::cuda::WarpReduceSum(qk);

      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0) {
        float score = qk * scale_log2;
        if (alibi_slope != 0.0f) {
          // Causal prefill: last position is (allowed_k_len - 1) for this
          // query.
          score +=
              (alibi_slope * static_cast<float>(kpos - (allowed_k_len - 1))) *
              kLog2e;
        }
        const float m_new = fmaxf(m, score);
        alpha = exp2f(m - m_new);
        beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;
      }
      alpha = op::paged_attention::cuda::WarpBroadcast(alpha, 0);
      beta = op::paged_attention::cuda::WarpBroadcast(beta, 0);

#if defined(__CUDA_ARCH__)
      if constexpr (std::is_same_v<TData, half>) {
        const int dim_base = lane * kDimsPerThread;
        const half2* v2 = reinterpret_cast<const half2*>(v_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 vf = __half22float2(v2[j]);
          acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
          acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
        }
      } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
        const int dim_base = lane * kDimsPerThread;
        const __nv_bfloat162* v2 =
            reinterpret_cast<const __nv_bfloat162*>(v_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 vf = __bfloat1622float2(v2[j]);
          acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
          acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
        }
      } else
#endif
      {
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
          const int dim = lane * kDimsPerThread + i;
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
  inv_l = op::paged_attention::cuda::WarpBroadcast(inv_l, 0);

#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    const float outval = acc[i] * inv_l;
    if (!is_active) {
      continue;
    }
    outptr[dim] = infini::ops::Caster<kDev>::template Cast<TData>(outval);
  }
}

// Pipelined CTA kernel (FA2-style): stage K/V loads with cp.async and overlap
// global->shared copies with compute.
//
// Design notes:
// - Keep shared memory <= 48KB for compatibility with multi-arch builds that
// include SM75.
// - Iterate by paged blocks (logical pages) so each tile stays within one
// physical block and
//   avoids per-token (page, off) mapping arrays in shared memory.
// - One warp computes one query token (same as warpcta kernels). Warps with
// shorter causal
//   limits simply mask the tail tokens but still participate in CTA-wide
//   barriers.
template <infini::ops::Device::Type kDev, typename TIndex, typename TData,
          int kHeadSize, int kBlockM, int kTokensPerTile, int kStages,
          typename TLengths>
__device__ void PagedAttentionPrefillWarpCtaKernelPipelined(
    TData* out, const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, TLengths total_kv_lens,
    const TIndex* cu_seqlens_q, const float* alibi_slopes, size_t num_kv_heads,
    float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t block_table_batch_stride, ptrdiff_t qstride,
    ptrdiff_t qhead_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride, ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride, ptrdiff_t o_stride, ptrdiff_t o_head_stride) {
  static_assert(kHeadSize == 64 || kHeadSize == 128 || kHeadSize == 192,
                "Only head_size 64/128/192 supported in v0.4.");
  static_assert(kBlockM > 0 && kBlockM <= 16, "kBlockM must be <= 16.");
  static_assert(kTokensPerTile == 32,
                "Pipelined CTA kernel currently assumes kTokensPerTile == 32.");
  static_assert(kStages >= 2 && kStages <= 3, "kStages must be 2 or 3.");
  static_assert(sizeof(TData) == 2,
                "Pipelined CTA kernel supports only fp16/bf16.");

  constexpr int kWarpSize = 32;
  static_assert(kHeadSize % kWarpSize == 0,
                "kHeadSize must be divisible by 32.");
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  if (warp_id >= kBlockM) {
    return;
  }

  const int head_idx = static_cast<int>(blockIdx.x);
  const int seqidx = static_cast<int>(blockIdx.y);
  const int m_block = static_cast<int>(blockIdx.z);

  const TIndex qstart = cu_seqlens_q[seqidx];
  const TIndex qend = cu_seqlens_q[seqidx + 1];
  const int qlen = static_cast<int>(qend - qstart);
  if (qlen <= 0) {
    return;
  }

  const int m_start = m_block * kBlockM;
  const int qtoken_local = m_start + warp_id;
  // Uniform return for empty tail CTAs (avoid deadlock with __syncthreads).
  if (m_start >= qlen) {
    return;
  }
  const bool is_active = (qtoken_local < qlen);

  const int kv_len_total = static_cast<int>(total_kv_lens[seqidx]);
  const int history_len = kv_len_total - qlen;
  const int allowed_k_len = is_active ? (history_len + qtoken_local + 1) : 0;

  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  int64_t qtoken = qstart;
  if (is_active) {
    qtoken += static_cast<int64_t>(qtoken_local);
  }

  const TIndex* block_table =
      block_tables + static_cast<int64_t>(seqidx) *
                         static_cast<int64_t>(block_table_batch_stride);

  const TData* qptr = nullptr;
  TData* outptr = nullptr;
  if (is_active) {
    qptr = q + qtoken * qstride + static_cast<int64_t>(head_idx) * qhead_stride;
    outptr = out + qtoken * o_stride +
             static_cast<int64_t>(head_idx) * o_head_stride;
  }

  float qreg[kDimsPerThread];
  float acc[kDimsPerThread];
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    qreg[i] = is_active ? static_cast<float>(qptr[dim]) : 0.0f;
    acc[i] = 0.0f;
  }

#if defined(__CUDA_ARCH__)
  float2 qreg2[kDimsPerThread / 2];
#pragma unroll
  for (int j = 0; j < kDimsPerThread / 2; ++j) {
    qreg2[j] = make_float2(qreg[j * 2 + 0], qreg[j * 2 + 1]);
  }
#endif

  float m = -INFINITY;
  float l = 0.0f;

  // For this CTA, scan KV up to the max causal limit among active warps.
  const int max_qin_tile = min(m_start + kBlockM, qlen);
  const int max_allowed_k_len = min(history_len + max_qin_tile, kv_len_total);
  if (max_allowed_k_len <= 0) {
    // Nothing to attend to (should be rare). Produce zeros.
    if (is_active) {
#pragma unroll
      for (int i = 0; i < kDimsPerThread; ++i) {
        const int dim = lane * kDimsPerThread + i;
        outptr[dim] = TData{};
      }
    }
    return;
  }

  // cp.async uses 16B chunks; for fp16/bf16 that's 8 elements.
  constexpr int kChunkElems = 8;
  constexpr int CHUNKS = kHeadSize / kChunkElems;
  constexpr int LOADS_PER_TILE = CHUNKS * kTokensPerTile;

  // Multi-stage pipeline buffers.
  __shared__ __align__(16) TData sh_k[kStages][kTokensPerTile][kHeadSize];
  __shared__ __align__(16) TData sh_v[kStages][kTokensPerTile][kHeadSize];
  // Per-warp scratch for tile-wise softmax (scores over kTokensPerTile).
  // We keep scores in shared so each lane can load its token score (lane ->
  // token index), then weights are broadcast via warp shuffles to avoid extra
  // shared-memory traffic.
  __shared__ float sh_scores[kBlockM][kTokensPerTile];
  // Store Q in shared (per warp). This enables more tile-level parallelism in
  // score computation without expensive cross-lane shuffles of Q registers.
  __shared__ __align__(16) TData sh_q[kBlockM][kHeadSize];

  const int pbs = static_cast<int>(page_block_size);
  const int tid = threadIdx.x;

  // Populate per-warp Q shared tile once.
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    sh_q[warp_id][dim] = is_active ? qptr[dim] : TData{};
  }
  __syncwarp();

  int t_base = 0;
  for (int logical_block = 0;
       t_base < max_allowed_k_len &&
       logical_block < static_cast<int>(max_num_blocks_per_seq);
       ++logical_block, t_base += pbs) {
    const int physical_block = static_cast<int>(block_table[logical_block]);

    const TData* k_base =
        k_cache + static_cast<int64_t>(physical_block) * k_batch_stride +
        static_cast<int64_t>(kv_head_idx) * k_head_stride;
    const TData* v_base =
        v_cache + static_cast<int64_t>(physical_block) * v_batch_stride +
        static_cast<int64_t>(kv_head_idx) * v_head_stride;

    const int token_end = min(pbs, max_allowed_k_len - t_base);
    const int num_tiles = (token_end + kTokensPerTile - 1) / kTokensPerTile;
    if (num_tiles <= 0) {
      continue;
    }

    int pending_groups = 0;
    const int preload = min(kStages, num_tiles);
    for (int ti = 0; ti < preload; ++ti) {
      const int token_in_block = ti * kTokensPerTile;
      const int tile_n = min(kTokensPerTile, token_end - token_in_block);
      for (int li = tid; li < LOADS_PER_TILE; li += blockDim.x) {
        const int tok = li / CHUNKS;
        const int chunk = li - tok * CHUNKS;
        const int off = chunk * kChunkElems;
        if (tok < tile_n) {
          const TData* k_src =
              k_base +
              static_cast<int64_t>(token_in_block + tok) * k_row_stride + off;
          const TData* v_src =
              v_base +
              static_cast<int64_t>(token_in_block + tok) * v_row_stride + off;
          op::paged_attention::cuda::CpAsyncCaSharedGlobal16(
              &sh_k[ti][tok][off], k_src);
          op::paged_attention::cuda::CpAsyncCaSharedGlobal16(
              &sh_v[ti][tok][off], v_src);
        } else {
          reinterpret_cast<uint4*>(&sh_k[ti][tok][off])[0] =
              make_uint4(0, 0, 0, 0);
          reinterpret_cast<uint4*>(&sh_v[ti][tok][off])[0] =
              make_uint4(0, 0, 0, 0);
        }
      }
      op::paged_attention::cuda::CpAsyncCommit();
      ++pending_groups;
    }

    int desired_pending = pending_groups - 1;
    if (desired_pending < 0) {
      desired_pending = 0;
    }
    if (desired_pending > (kStages - 1)) {
      desired_pending = (kStages - 1);
    }
    op::paged_attention::cuda::CpAsyncWaitGroupRt(desired_pending);
    pending_groups = desired_pending;
    __syncthreads();

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      const int buf = tile_idx % kStages;
      const int token_in_block = tile_idx * kTokensPerTile;
      const int tile_n = min(kTokensPerTile, token_end - token_in_block);

      const int global_k_base = t_base + token_in_block;
      // Tile-wise online softmax (more FA2-like than per-token update):
      // 1) Compute scores for this tile (masked to each warp's causal limit).
      // 2) Compute tile max + sumexp.
      // 3) Accumulate weighted V for the tile.
      // 4) Merge into running (m, l, acc) in a numerically stable way.
      //
      // NOTE: this does not yet implement MMA / full tile-level GEMM; it mainly
      // reduces the serial (lane0) online-softmax update frequency from
      // per-token to per-tile.
      float alpha = 1.0f;
      float beta = 0.0f;
      float tile_sumexp = 0.0f;
      float tile_m = -INFINITY;

      if (allowed_k_len > 0) {
        // 1) scores
        // Increase tile-level parallelism vs the previous per-token loop:
        // split the warp into 4 groups of 8 lanes; each group computes one
        // token score in parallel.
        constexpr int LANES_PER_GROUP = 8;
        constexpr int GROUPS_PER_WARP = 4;
        constexpr int DIMS_PER_GROUP_LANE = kHeadSize / LANES_PER_GROUP;
        static_assert(kHeadSize % LANES_PER_GROUP == 0,
                      "kHeadSize must be divisible by 8.");

        const int group_id = lane / LANES_PER_GROUP;      // [0..3]
        const int lane_g = lane & (LANES_PER_GROUP - 1);  // [0..7]
        const unsigned int group_mask = 0xFFu << (group_id * LANES_PER_GROUP);

        for (int j_base = 0; j_base < kTokensPerTile;
             j_base += GROUPS_PER_WARP) {
          const int j = j_base + group_id;  // token index in [0..31]
          const int kpos = global_k_base + j;

          const bool token_in_tile = (j < tile_n);
          const bool token_unmasked = token_in_tile && (kpos < allowed_k_len);

          float qk_part = 0.0f;
          if (token_unmasked) {
            const TData* k_ptr = &sh_k[buf][j][0];
            const int dim_base = lane_g * DIMS_PER_GROUP_LANE;
#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<TData, half>) {
              const half2* q2 =
                  reinterpret_cast<const half2*>(&sh_q[warp_id][dim_base]);
              const half2* k2 =
                  reinterpret_cast<const half2*>(k_ptr + dim_base);
#pragma unroll
              for (int t = 0; t < DIMS_PER_GROUP_LANE / 2; ++t) {
                const float2 qf = __half22float2(q2[t]);
                const float2 kf = __half22float2(k2[t]);
                qk_part += qf.x * kf.x + qf.y * kf.y;
              }
            } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
              const __nv_bfloat162* q2 =
                  reinterpret_cast<const __nv_bfloat162*>(
                      &sh_q[warp_id][dim_base]);
              const __nv_bfloat162* k2 =
                  reinterpret_cast<const __nv_bfloat162*>(k_ptr + dim_base);
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
                qk_part += static_cast<float>(sh_q[warp_id][dim_base + t]) *
                           static_cast<float>(k_ptr[dim_base + t]);
              }
            }
          }

          // Reduce within 8-lane group.
          for (int offset = LANES_PER_GROUP / 2; offset > 0; offset >>= 1) {
            qk_part +=
                __shfl_down_sync(group_mask, qk_part, offset, LANES_PER_GROUP);
          }

          if (lane_g == 0) {
            float score = -INFINITY;
            if (token_unmasked) {
              score = qk_part * scale_log2;
              if (alibi_slope != 0.0f) {
                const int causal_limit = allowed_k_len - 1;
                score +=
                    (alibi_slope * static_cast<float>(kpos - causal_limit)) *
                    kLog2e;
              }
            }
            sh_scores[warp_id][j] = score;
          }
        }
        __syncwarp();

        // 2) tile max + sumexp (lane t corresponds to token t within the tile)
        const float score_lane =
            (lane < tile_n) ? sh_scores[warp_id][lane] : -INFINITY;
        float tile_m_tmp = op::paged_attention::cuda::WarpReduceMax(score_lane);
        tile_m_tmp = __shfl_sync(0xffffffff, tile_m_tmp, 0);
        tile_m = tile_m_tmp;

        float w_lane = 0.0f;
        if (lane < tile_n && tile_m != -INFINITY) {
          w_lane = exp2f(score_lane - tile_m);
        }
        float sumexp_tmp = op::paged_attention::cuda::WarpReduceSum(w_lane);
        sumexp_tmp = __shfl_sync(0xffffffff, sumexp_tmp, 0);
        tile_sumexp = sumexp_tmp;

        // 3) weighted V for this tile (per lane owns kHeadSize/32 dims)
        float acc_tile[kDimsPerThread];
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
          acc_tile[i] = 0.0f;
        }

        if (tile_sumexp > 0.0f) {
          for (int j = 0; j < tile_n; ++j) {
            // Broadcast weight for token j from lane j.
            const float wj = __shfl_sync(0xffffffff, w_lane, j);
            const TData* v_ptr = &sh_v[buf][j][0];
#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<TData, half>) {
              const int dim_base = lane * kDimsPerThread;
              const half2* v2 =
                  reinterpret_cast<const half2*>(v_ptr + dim_base);
#pragma unroll
              for (int jj = 0; jj < kDimsPerThread / 2; ++jj) {
                const float2 vf = __half22float2(v2[jj]);
                acc_tile[jj * 2 + 0] += wj * vf.x;
                acc_tile[jj * 2 + 1] += wj * vf.y;
              }
            } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
              const int dim_base = lane * kDimsPerThread;
              const __nv_bfloat162* v2 =
                  reinterpret_cast<const __nv_bfloat162*>(v_ptr + dim_base);
#pragma unroll
              for (int jj = 0; jj < kDimsPerThread / 2; ++jj) {
                const float2 vf = __bfloat1622float2(v2[jj]);
                acc_tile[jj * 2 + 0] += wj * vf.x;
                acc_tile[jj * 2 + 1] += wj * vf.y;
              }
            } else
#endif
            {
#pragma unroll
              for (int i = 0; i < kDimsPerThread; ++i) {
                const int dim = lane * kDimsPerThread + i;
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
        for (int i = 0; i < kDimsPerThread; ++i) {
          acc[i] = acc[i] * alpha + beta * acc_tile[i];
        }
      }

      // IMPORTANT: warps in this CTA can have different allowed_k_len (due to
      // causal mask + history), so they may finish the token loop at different
      // times. We must not start prefetching into the circular shared-memory
      // buffer until all warps finish consuming the current tile.
      __syncthreads();

      // Prefetch the tile that will reuse this buffer (kStages steps ahead).
      const int prefetch_tile = tile_idx + kStages;
      if (prefetch_tile < num_tiles) {
        const int token_prefetch = prefetch_tile * kTokensPerTile;
        const int prefetch_n = min(kTokensPerTile, token_end - token_prefetch);
        for (int li = tid; li < LOADS_PER_TILE; li += blockDim.x) {
          const int tok = li / CHUNKS;
          const int chunk = li - tok * CHUNKS;
          const int off = chunk * kChunkElems;
          if (tok < prefetch_n) {
            const TData* k_src =
                k_base +
                static_cast<int64_t>(token_prefetch + tok) * k_row_stride + off;
            const TData* v_src =
                v_base +
                static_cast<int64_t>(token_prefetch + tok) * v_row_stride + off;
            op::paged_attention::cuda::CpAsyncCaSharedGlobal16(
                &sh_k[buf][tok][off], k_src);
            op::paged_attention::cuda::CpAsyncCaSharedGlobal16(
                &sh_v[buf][tok][off], v_src);
          } else {
            reinterpret_cast<uint4*>(&sh_k[buf][tok][off])[0] =
                make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(&sh_v[buf][tok][off])[0] =
                make_uint4(0, 0, 0, 0);
          }
        }
        op::paged_attention::cuda::CpAsyncCommit();
        ++pending_groups;
      }

      if (tile_idx + 1 < num_tiles) {
        int desired_pending2 = pending_groups - 1;
        if (desired_pending2 < 0) {
          desired_pending2 = 0;
        }
        if (desired_pending2 > (kStages - 1)) {
          desired_pending2 = (kStages - 1);
        }
        op::paged_attention::cuda::CpAsyncWaitGroupRt(desired_pending2);
        pending_groups = desired_pending2;
        __syncthreads();
      }
    }

    op::paged_attention::cuda::CpAsyncWaitAll();
    __syncthreads();
  }

  float inv_l = 0.0f;
  if (lane == 0) {
    inv_l = 1.0f / (l + 1e-6f);
  }
  inv_l = op::paged_attention::cuda::WarpBroadcast(inv_l, 0);

#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    const float outval = acc[i] * inv_l;
    if (!is_active) {
      continue;
    }
    outptr[dim] = infini::ops::Caster<kDev>::template Cast<TData>(outval);
  }
}

// Split-KV prefill (FA2-style): each split scans a shard of KV and writes
// partial (m, l, acc) to workspace. A separate combine kernel merges splits
// into the final output.
//
// Notes:
// - Implemented for the pipelined CTA kernel family (warpcta8pipe). We split by
// logical paged blocks.
// - Each warp still applies its own causal limit (allowed_k_len) so correctness
// is preserved.
template <typename TIndex, typename TData, int kHeadSize, int kBlockM,
          int kTokensPerTile, int kStages>
__device__ void PagedAttentionPrefillWarpCtaKernelPipelinedSplitKv(
    float* partial_acc,  // [num_splits, total_qtokens, num_heads, head_size]
    float* partial_m,    // [num_splits, total_qtokens, num_heads]
    float* partial_l,    // [num_splits, total_qtokens, num_heads]
    int split_idx, int num_splits, int m_block, size_t total_qtokens,
    const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, const TIndex* total_kv_lens,
    const TIndex* cu_seqlens_q, const float* alibi_slopes, size_t num_kv_heads,
    float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t block_table_batch_stride, ptrdiff_t qstride,
    ptrdiff_t qhead_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride, ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride) {
  (void)max_num_blocks_per_seq;

  static_assert(kHeadSize == 64 || kHeadSize == 128 || kHeadSize == 192,
                "Only head_size 64/128/192 supported in v0.4.");
  static_assert(kBlockM > 0 && kBlockM <= 16, "kBlockM must be <= 16.");
  static_assert(kTokensPerTile == 32,
                "Split-KV prefill assumes kTokensPerTile == 32.");
  static_assert(kStages >= 2 && kStages <= 3, "kStages must be 2 or 3.");
  static_assert(sizeof(TData) == 2,
                "Split-KV prefill supports only fp16/bf16.");

  constexpr int kWarpSize = 32;
  static_assert(kHeadSize % kWarpSize == 0,
                "kHeadSize must be divisible by 32.");
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  if (warp_id >= kBlockM) {
    return;
  }

  const int head_idx = static_cast<int>(blockIdx.x);
  const int seqidx = static_cast<int>(blockIdx.y);

  const TIndex qstart = cu_seqlens_q[seqidx];
  const TIndex qend = cu_seqlens_q[seqidx + 1];
  const int qlen = static_cast<int>(qend - qstart);
  if (qlen <= 0) {
    return;
  }

  const int m_start = m_block * kBlockM;
  const int qtoken_local = m_start + warp_id;
  if (m_start >= qlen) {
    return;  // uniform
  }
  const bool is_active = (qtoken_local < qlen);

  const int kv_len_total = static_cast<int>(total_kv_lens[seqidx]);
  const int history_len = kv_len_total - qlen;
  const int allowed_k_len = is_active ? (history_len + qtoken_local + 1) : 0;

  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  int64_t qtoken = qstart;
  if (is_active) {
    qtoken += static_cast<int64_t>(qtoken_local);
  }

  const size_t n = total_qtokens * static_cast<size_t>(num_heads);
  size_t base = 0;
  if (is_active) {
    base = static_cast<size_t>(qtoken) * static_cast<size_t>(num_heads) +
           static_cast<size_t>(head_idx);
  }

  const TIndex* block_table =
      block_tables + static_cast<int64_t>(seqidx) *
                         static_cast<int64_t>(block_table_batch_stride);
  const TData* qptr = nullptr;
  if (is_active) {
    qptr = q + qtoken * qstride + static_cast<int64_t>(head_idx) * qhead_stride;
  }

  float qreg[kDimsPerThread];
  float acc[kDimsPerThread];
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    qreg[i] = is_active ? static_cast<float>(qptr[dim]) : 0.0f;
    acc[i] = 0.0f;
  }

  float m = -INFINITY;
  float l = 0.0f;

  const int max_qin_tile = min(m_start + kBlockM, qlen);
  const int max_allowed_k_len = min(history_len + max_qin_tile, kv_len_total);
  if (max_allowed_k_len <= 0) {
    if (is_active) {
      const size_t idx = static_cast<size_t>(split_idx) * n + base;
      if (lane == 0) {
        partial_m[idx] = -INFINITY;
        partial_l[idx] = 0.0f;
      }
#pragma unroll
      for (int i = 0; i < kDimsPerThread; ++i) {
        const int dim = lane * kDimsPerThread + i;
        partial_acc[idx * kHeadSize + dim] = 0.0f;
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
      for (int i = 0; i < kDimsPerThread; ++i) {
        const int dim = lane * kDimsPerThread + i;
        partial_acc[idx * kHeadSize + dim] = 0.0f;
      }
    }
    return;
  }

  const int max_allowed_k_len_split = min(max_allowed_k_len, end_block * pbs);

  constexpr int kChunkElems = 8;
  constexpr int CHUNKS = kHeadSize / kChunkElems;
  constexpr int LOADS_PER_TILE = CHUNKS * kTokensPerTile;

  __shared__ __align__(16) TData sh_k[kStages][kTokensPerTile][kHeadSize];
  __shared__ __align__(16) TData sh_v[kStages][kTokensPerTile][kHeadSize];
  __shared__ float sh_scores[kBlockM][kTokensPerTile];

  const int tid = threadIdx.x;

  int t_base = start_block * pbs;
  for (int logical_block = start_block; t_base < max_allowed_k_len_split;
       ++logical_block, t_base += pbs) {
    const int physical_block = static_cast<int>(block_table[logical_block]);

    const TData* k_base =
        k_cache + static_cast<int64_t>(physical_block) * k_batch_stride +
        static_cast<int64_t>(kv_head_idx) * k_head_stride;
    const TData* v_base =
        v_cache + static_cast<int64_t>(physical_block) * v_batch_stride +
        static_cast<int64_t>(kv_head_idx) * v_head_stride;

    const int token_end = min(pbs, max_allowed_k_len_split - t_base);
    const int num_tiles = (token_end + kTokensPerTile - 1) / kTokensPerTile;
    if (num_tiles <= 0) {
      continue;
    }

    int pending_groups = 0;
    const int preload = min(kStages, num_tiles);
    for (int ti = 0; ti < preload; ++ti) {
      const int token_in_block = ti * kTokensPerTile;
      const int tile_n = min(kTokensPerTile, token_end - token_in_block);
      for (int li = tid; li < LOADS_PER_TILE; li += blockDim.x) {
        const int tok = li / CHUNKS;
        const int chunk = li - tok * CHUNKS;
        const int off = chunk * kChunkElems;
        if (tok < tile_n) {
          const TData* k_src =
              k_base +
              static_cast<int64_t>(token_in_block + tok) * k_row_stride + off;
          const TData* v_src =
              v_base +
              static_cast<int64_t>(token_in_block + tok) * v_row_stride + off;
          op::paged_attention::cuda::CpAsyncCaSharedGlobal16(
              &sh_k[ti][tok][off], k_src);
          op::paged_attention::cuda::CpAsyncCaSharedGlobal16(
              &sh_v[ti][tok][off], v_src);
        } else {
          reinterpret_cast<uint4*>(&sh_k[ti][tok][off])[0] =
              make_uint4(0, 0, 0, 0);
          reinterpret_cast<uint4*>(&sh_v[ti][tok][off])[0] =
              make_uint4(0, 0, 0, 0);
        }
      }
      op::paged_attention::cuda::CpAsyncCommit();
      ++pending_groups;
    }

    int desired_pending = pending_groups - 1;
    if (desired_pending < 0) {
      desired_pending = 0;
    }
    if (desired_pending > (kStages - 1)) {
      desired_pending = (kStages - 1);
    }
    op::paged_attention::cuda::CpAsyncWaitGroupRt(desired_pending);
    pending_groups = desired_pending;
    __syncthreads();

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      const int buf = tile_idx % kStages;
      const int token_in_block = tile_idx * kTokensPerTile;
      const int tile_n = min(kTokensPerTile, token_end - token_in_block);
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
            const TData* k_ptr = &sh_k[buf][j][0];
#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<TData, half>) {
              const int dim_base = lane * kDimsPerThread;
              const half2* q2 = reinterpret_cast<const half2*>(qptr + dim_base);
              const half2* k2 =
                  reinterpret_cast<const half2*>(k_ptr + dim_base);
#pragma unroll
              for (int ii = 0; ii < kDimsPerThread / 2; ++ii) {
                const float2 qf = __half22float2(q2[ii]);
                const float2 kf = __half22float2(k2[ii]);
                qk = fmaf(qf.x, kf.x, qk);
                qk = fmaf(qf.y, kf.y, qk);
              }
            } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
              const int dim_base = lane * kDimsPerThread;
              const __nv_bfloat162* q2 =
                  reinterpret_cast<const __nv_bfloat162*>(qptr + dim_base);
              const __nv_bfloat162* k2 =
                  reinterpret_cast<const __nv_bfloat162*>(k_ptr + dim_base);
#pragma unroll
              for (int ii = 0; ii < kDimsPerThread / 2; ++ii) {
                const float2 qf = __bfloat1622float2(q2[ii]);
                const float2 kf = __bfloat1622float2(k2[ii]);
                qk = fmaf(qf.x, kf.x, qk);
                qk = fmaf(qf.y, kf.y, qk);
              }
            } else
#endif
            {
#pragma unroll
              for (int i = 0; i < kDimsPerThread; ++i) {
                const int dim = lane * kDimsPerThread + i;
                qk = fmaf(qreg[i], static_cast<float>(k_ptr[dim]), qk);
              }
            }
          }
          qk = op::paged_attention::cuda::WarpReduceSum(qk);
          if (lane == 0) {
            float score = token_unmasked ? (qk * scale_log2) : -INFINITY;
            if (token_unmasked && alibi_slope != 0.0f) {
              const int causal_limit = allowed_k_len - 1;
              score += (alibi_slope * static_cast<float>(kpos - causal_limit)) *
                       kLog2e;
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
        max_tmp = op::paged_attention::cuda::WarpReduceMax(max_tmp);
        max_tmp = __shfl_sync(0xffffffff, max_tmp, 0);
        tile_m = max_tmp;

        if (lane < tile_n) {
          const float s = sh_scores[warp_id][lane];
          w_lane = (s == -INFINITY) ? 0.0f : exp2f(s - tile_m);
        } else {
          w_lane = 0.0f;
        }
        float sumexp_tmp = op::paged_attention::cuda::WarpReduceSum(w_lane);
        sumexp_tmp = __shfl_sync(0xffffffff, sumexp_tmp, 0);
        tile_sumexp = sumexp_tmp;

        // 3) weighted V for this tile
        float acc_tile[kDimsPerThread];
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
          acc_tile[i] = 0.0f;
        }
        if (tile_sumexp > 0.0f) {
          for (int j = 0; j < tile_n; ++j) {
            const float wj = __shfl_sync(0xffffffff, w_lane, j);
            const TData* v_ptr = &sh_v[buf][j][0];
#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<TData, half>) {
              const int dim_base = lane * kDimsPerThread;
              const half2* v2 =
                  reinterpret_cast<const half2*>(v_ptr + dim_base);
#pragma unroll
              for (int jj = 0; jj < kDimsPerThread / 2; ++jj) {
                const float2 vf = __half22float2(v2[jj]);
                acc_tile[jj * 2 + 0] += wj * vf.x;
                acc_tile[jj * 2 + 1] += wj * vf.y;
              }
            } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
              const int dim_base = lane * kDimsPerThread;
              const __nv_bfloat162* v2 =
                  reinterpret_cast<const __nv_bfloat162*>(v_ptr + dim_base);
#pragma unroll
              for (int jj = 0; jj < kDimsPerThread / 2; ++jj) {
                const float2 vf = __bfloat1622float2(v2[jj]);
                acc_tile[jj * 2 + 0] += wj * vf.x;
                acc_tile[jj * 2 + 1] += wj * vf.y;
              }
            } else
#endif
            {
#pragma unroll
              for (int i = 0; i < kDimsPerThread; ++i) {
                const int dim = lane * kDimsPerThread + i;
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
        for (int i = 0; i < kDimsPerThread; ++i) {
          acc[i] = acc[i] * alpha + beta * acc_tile[i];
        }
      }

      __syncthreads();

      const int prefetch_tile = tile_idx + kStages;
      if (prefetch_tile < num_tiles) {
        const int token_prefetch = prefetch_tile * kTokensPerTile;
        const int prefetch_n = min(kTokensPerTile, token_end - token_prefetch);
        for (int li = tid; li < LOADS_PER_TILE; li += blockDim.x) {
          const int tok = li / CHUNKS;
          const int chunk = li - tok * CHUNKS;
          const int off = chunk * kChunkElems;
          if (tok < prefetch_n) {
            const TData* k_src =
                k_base +
                static_cast<int64_t>(token_prefetch + tok) * k_row_stride + off;
            const TData* v_src =
                v_base +
                static_cast<int64_t>(token_prefetch + tok) * v_row_stride + off;
            op::paged_attention::cuda::CpAsyncCaSharedGlobal16(
                &sh_k[buf][tok][off], k_src);
            op::paged_attention::cuda::CpAsyncCaSharedGlobal16(
                &sh_v[buf][tok][off], v_src);
          } else {
            reinterpret_cast<uint4*>(&sh_k[buf][tok][off])[0] =
                make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(&sh_v[buf][tok][off])[0] =
                make_uint4(0, 0, 0, 0);
          }
        }
        op::paged_attention::cuda::CpAsyncCommit();
        ++pending_groups;
      }

      if (tile_idx + 1 < num_tiles) {
        int desired_pending2 = pending_groups - 1;
        if (desired_pending2 < 0) {
          desired_pending2 = 0;
        }
        if (desired_pending2 > (kStages - 1)) {
          desired_pending2 = (kStages - 1);
        }
        op::paged_attention::cuda::CpAsyncWaitGroupRt(desired_pending2);
        pending_groups = desired_pending2;
        __syncthreads();
      }
    }

    op::paged_attention::cuda::CpAsyncWaitAll();
    __syncthreads();
  }

  if (is_active) {
    const size_t idx = static_cast<size_t>(split_idx) * n + base;
    if (lane == 0) {
      partial_m[idx] = m;
      partial_l[idx] = l;
    }
#pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
      const int dim = lane * kDimsPerThread + i;
      partial_acc[idx * kHeadSize + dim] = acc[i];
    }
  }
}

template <infini::ops::Device::Type kDev, typename TData, int kHeadSize>
__device__ void PagedAttentionPrefillSplitKvCombineWarpKernel(
    TData* out,
    const float*
        partial_acc,  // [num_splits, total_qtokens, num_heads, head_size]
    const float* partial_m,  // [num_splits, total_qtokens, num_heads]
    const float* partial_l,  // [num_splits, total_qtokens, num_heads]
    int num_splits, size_t total_qtokens, ptrdiff_t o_stride,
    ptrdiff_t o_head_stride) {
  const int head_idx = static_cast<int>(blockIdx.x);
  const int token_idx = static_cast<int>(blockIdx.y);
  const int lane = threadIdx.x;
  constexpr int kWarpSize = 32;
  static_assert(kHeadSize % kWarpSize == 0,
                "kHeadSize must be divisible by 32.");
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;

  const int num_heads = gridDim.x;
  const size_t n = total_qtokens * static_cast<size_t>(num_heads);
  const size_t base =
      static_cast<size_t>(token_idx) * static_cast<size_t>(num_heads) +
      static_cast<size_t>(head_idx);

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

  TData* outptr = out + static_cast<int64_t>(token_idx) * o_stride +
                  static_cast<int64_t>(head_idx) * o_head_stride;
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      const float ms = partial_m[static_cast<size_t>(s) * n + base];
      const float w = exp2f(ms - m);
      acc +=
          partial_acc[(static_cast<size_t>(s) * n + base) * kHeadSize + dim] *
          w;
    }
    const float o = acc * inv_l;
    outptr[dim] = infini::ops::Caster<kDev>::template Cast<TData>(o);
  }
}

// Variant for large K tile where (K+V) shared memory would exceed the per-block
// limit on some GPUs. We keep K in shared memory for reuse across warps, but
// load V directly from global memory.
template <infini::ops::Device::Type kDev, typename TIndex, typename TData,
          int kHeadSize, int kBlockM, int kBlockN>
__device__ void PagedAttentionPrefillWarpCtaKernelKOnly(
    TData* out, const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, const TIndex* total_kv_lens,
    const TIndex* cu_seqlens_q, const float* alibi_slopes, size_t num_kv_heads,
    float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t block_table_batch_stride, ptrdiff_t qstride,
    ptrdiff_t qhead_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride, ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride, ptrdiff_t o_stride, ptrdiff_t o_head_stride) {
  static_assert(kHeadSize == 64 || kHeadSize == 128 || kHeadSize == 192,
                "Only head_size 64/128/192 supported in v0.4.");
  static_assert(kBlockM > 0 && kBlockM <= 16, "kBlockM must be <=16.");
  static_assert(kBlockN > 0 && kBlockN <= 128, "kBlockN must be <=128.");

  constexpr int kWarpSize = 32;
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;
  static_assert(kHeadSize % kWarpSize == 0,
                "kHeadSize must be divisible by 32.");

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  if (warp_id >= kBlockM) {
    return;
  }

  const int head_idx = static_cast<int>(blockIdx.x);
  const int seqidx = static_cast<int>(blockIdx.y);
  const int m_block = static_cast<int>(blockIdx.z);

  const TIndex qstart = cu_seqlens_q[seqidx];
  const TIndex qend = cu_seqlens_q[seqidx + 1];
  const int qlen = static_cast<int>(qend - qstart);
  if (qlen <= 0) {
    return;
  }

  const int m_start = m_block * kBlockM;
  const int qtoken_local = m_start + warp_id;
  // IMPORTANT: do not early-return for a subset of warps in this CTA because we
  // use __syncthreads() later. Tail tiles are handled by masking inactive
  // warps.
  if (m_start >= qlen) {
    return;  // uniform across the CTA
  }
  const bool is_active = (qtoken_local < qlen);

  const int kv_len_total = static_cast<int>(total_kv_lens[seqidx]);
  const int history_len = kv_len_total - qlen;
  const int allowed_k_len = is_active ? (history_len + qtoken_local + 1) : 0;

  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  int64_t qtoken = qstart;
  if (is_active) {
    qtoken += static_cast<int64_t>(qtoken_local);
  }

  const TIndex* block_table =
      block_tables + static_cast<int64_t>(seqidx) *
                         static_cast<int64_t>(block_table_batch_stride);

  const TData* qptr = nullptr;
  TData* outptr = nullptr;
  if (is_active) {
    qptr = q + qtoken * qstride + static_cast<int64_t>(head_idx) * qhead_stride;
    outptr = out + qtoken * o_stride +
             static_cast<int64_t>(head_idx) * o_head_stride;
  }

  float qreg[kDimsPerThread];
  float acc[kDimsPerThread];
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    qreg[i] = is_active ? static_cast<float>(qptr[dim]) : 0.0f;
    acc[i] = 0.0f;
  }

#if defined(__CUDA_ARCH__)
  float2 qreg2[kDimsPerThread / 2];
#pragma unroll
  for (int j = 0; j < kDimsPerThread / 2; ++j) {
    qreg2[j] = make_float2(qreg[j * 2 + 0], qreg[j * 2 + 1]);
  }
#endif

  float m = -INFINITY;
  float l = 0.0f;

  const int max_qin_tile = min(m_start + kBlockM, qlen);
  const int max_allowed_k_len = min(history_len + max_qin_tile, kv_len_total);

  __shared__ int32_t s_phys[kBlockN];
  __shared__ int32_t s_off[kBlockN];
  __shared__ __align__(16) TData s_k[kBlockN * kHeadSize];

  const int pbs = static_cast<int>(page_block_size);

  for (int k_base = 0; k_base < max_allowed_k_len; k_base += kBlockN) {
    const int tile_n = min(kBlockN, max_allowed_k_len - k_base);

    for (int t = threadIdx.x; t < tile_n; t += blockDim.x) {
      const int kpos = k_base + t;
      const int page = (pbs == 256) ? (kpos >> 8) : (kpos / pbs);
      const int off = (pbs == 256) ? (kpos & 255) : (kpos - page * pbs);
      const int32_t phys = static_cast<int32_t>(block_table[page]);
      s_phys[t] = phys;
      s_off[t] = off;
    }
    __syncthreads();

    const int tile_elems = tile_n * kHeadSize;
    for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
      const int t = idx / kHeadSize;
      const int dim = idx - t * kHeadSize;
      const int32_t phys = s_phys[t];
      const int32_t off = s_off[t];
      const TData* k_base_ptr =
          k_cache + static_cast<int64_t>(phys) * k_batch_stride +
          static_cast<int64_t>(off) * k_row_stride +
          static_cast<int64_t>(kv_head_idx) * k_head_stride;
      s_k[t * kHeadSize + dim] = k_base_ptr[dim];
    }
    __syncthreads();

    for (int t = 0; t < tile_n; ++t) {
      const int kpos = k_base + t;
      if (kpos >= allowed_k_len) {
        break;
      }
      const TData* k_ptr = s_k + t * kHeadSize;
      const int32_t phys = s_phys[t];
      const int32_t off = s_off[t];
      const TData* v_ptr = v_cache +
                           static_cast<int64_t>(phys) * v_batch_stride +
                           static_cast<int64_t>(off) * v_row_stride +
                           static_cast<int64_t>(kv_head_idx) * v_head_stride;

      float qk = 0.0f;
#if defined(__CUDA_ARCH__)
      if constexpr (std::is_same_v<TData, half>) {
        const int dim_base = lane * kDimsPerThread;
        const half2* k2 = reinterpret_cast<const half2*>(k_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 qf = qreg2[j];
          const float2 kf = __half22float2(k2[j]);
          qk += qf.x * kf.x + qf.y * kf.y;
        }
      } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
        const int dim_base = lane * kDimsPerThread;
        const __nv_bfloat162* k2 =
            reinterpret_cast<const __nv_bfloat162*>(k_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 qf = qreg2[j];
          const float2 kf = __bfloat1622float2(k2[j]);
          qk += qf.x * kf.x + qf.y * kf.y;
        }
      } else
#endif
      {
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
          const int dim = lane * kDimsPerThread + i;
          qk += qreg[i] * static_cast<float>(k_ptr[dim]);
        }
      }

      qk = op::paged_attention::cuda::WarpReduceSum(qk);

      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0) {
        float score = qk * scale_log2;
        if (alibi_slope != 0.0f) {
          score +=
              (alibi_slope * static_cast<float>(kpos - (allowed_k_len - 1))) *
              kLog2e;
        }
        const float m_new = fmaxf(m, score);
        alpha = exp2f(m - m_new);
        beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;
      }
      alpha = op::paged_attention::cuda::WarpBroadcast(alpha, 0);
      beta = op::paged_attention::cuda::WarpBroadcast(beta, 0);

#if defined(__CUDA_ARCH__)
      if constexpr (std::is_same_v<TData, half>) {
        const int dim_base = lane * kDimsPerThread;
        const half2* v2 = reinterpret_cast<const half2*>(v_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 vf = __half22float2(v2[j]);
          acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
          acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
        }
      } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
        const int dim_base = lane * kDimsPerThread;
        const __nv_bfloat162* v2 =
            reinterpret_cast<const __nv_bfloat162*>(v_ptr + dim_base);
#pragma unroll
        for (int j = 0; j < kDimsPerThread / 2; ++j) {
          const float2 vf = __bfloat1622float2(v2[j]);
          acc[j * 2 + 0] = acc[j * 2 + 0] * alpha + beta * vf.x;
          acc[j * 2 + 1] = acc[j * 2 + 1] * alpha + beta * vf.y;
        }
      } else
#endif
      {
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
          const int dim = lane * kDimsPerThread + i;
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
  inv_l = op::paged_attention::cuda::WarpBroadcast(inv_l, 0);

#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    const float outval = acc[i] * inv_l;
    if (!is_active) {
      continue;
    }
    outptr[dim] = infini::ops::Caster<kDev>::template Cast<TData>(outval);
  }
}

// TensorCore (WMMA) score kernel (v0.4 experimental):
// - Target shape: head_dim=128, page_block_size=256, fp16.
// - Compute QK^T with WMMA into shared memory, then reuse the existing
// online-softmax + V accumulation
//   pattern (SIMT) per query row.
//
// Notes:
// - This is a correctness-first kernel. It doesn't yet use MMA for PV (P * V)
// update.
// - We keep the same grid mapping as other prefill kernels: blockIdx = (head,
// seq, m_block).
#if !defined(ENABLE_HYGON_API)
template <int kWarpSize, int kBlockN, int kHeadDim, int kDimsPerThread>
__device__ __forceinline__ void PagedAttentionPrefillMmaScoreUpdateRow(
    int lane, int k_base, int allowed_k_len,
    const float* scores_row,  // [kBlockN]
    const half* v_tile,       // [kBlockN, kHeadDim]
    float scale_log2, float alibi_slope_log2, float& m, float& l,
    float* acc) {  // [kDimsPerThread]

  // Max over keys in this tile.
  float local_max = -INFINITY;
  for (int t = lane; t < kBlockN; t += kWarpSize) {
    const int kpos = k_base + t;
    if (kpos >= allowed_k_len) {
      continue;
    }
    float score = scores_row[t] * scale_log2;
    if (alibi_slope_log2 != 0.0f) {
      score +=
          alibi_slope_log2 * static_cast<float>(kpos - (allowed_k_len - 1));
    }
    local_max = fmaxf(local_max, score);
  }
  float tile_m = op::paged_attention::cuda::WarpReduceMax(local_max);
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
        score +=
            alibi_slope_log2 * static_cast<float>(kpos - (allowed_k_len - 1));
      }
      const float w = exp2f(score - tile_m);
      sumexp_lane += w;

      const half* v_ptr = v_tile + t * kHeadDim + dim_base;
      const half2* v2 = reinterpret_cast<const half2*>(v_ptr);
#pragma unroll
      for (int j = 0; j < kDimsPerThread / 2; ++j) {
        const float2 vf = __half22float2(v2[j]);
        acc_tile[j * 2 + 0] += w * vf.x;
        acc_tile[j * 2 + 1] += w * vf.y;
      }
    }
  }

  float tile_sumexp = op::paged_attention::cuda::WarpReduceSum(sumexp_lane);
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

template <typename TIndex, int kWarpSize, int kHeadDim, int kDimsPerThread>
__device__ __forceinline__ void PagedAttentionPrefillMmaScoreWriteRow(
    int lane, bool active, int qtoken_local, TIndex qstart, int head_idx,
    half* out, ptrdiff_t o_stride, ptrdiff_t o_head_stride, float l,
    const float* acc) {  // [kDimsPerThread]
  if (!active) {
    return;
  }

  float inv_l = 0.0f;
  if (lane == 0) {
    inv_l = 1.0f / (l + 1e-6f);
  }
  inv_l = op::paged_attention::cuda::WarpBroadcast(inv_l, 0);

  const int64_t qtoken = qstart + static_cast<int64_t>(qtoken_local);
  half* outptr =
      out + qtoken * o_stride + static_cast<int64_t>(head_idx) * o_head_stride;
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    outptr[dim] = __float2half_rn(acc[i] * inv_l);
  }
}

template <typename TIndex>
__device__ void PagedAttentionPrefillWarpCta8MmaHd128Kernel(
    half* out, const half* q, const half* k_cache, const half* v_cache,
    const TIndex* block_tables, const TIndex* total_kv_lens,
    const TIndex* cu_seqlens_q, const float* alibi_slopes, size_t num_kv_heads,
    float scale, size_t max_num_blocks_per_seq, size_t page_block_size,
    ptrdiff_t block_table_batch_stride, ptrdiff_t qstride,
    ptrdiff_t qhead_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride, ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride, ptrdiff_t o_stride, ptrdiff_t o_head_stride) {
  (void)max_num_blocks_per_seq;

  constexpr int kWarpSize = 32;
  constexpr int kWarps = 8;
  constexpr int kHeadDim = 128;
  // Extra padding in the K dimension to reduce shared-memory bank conflicts for
  // ldmatrix / wmma loads. NOTE: FA2 uses a swizzled smem layout; padding is a
  // smaller step that keeps our code simple.
  constexpr int kHeadDimSmem =
      136;  // must be a multiple of 8 for wmma::load_matrix_sync
  constexpr int kBlockM = 16;  // 2 rows per warp
  // Keep static shared memory <= 48KB for compatibility with build targets that
  // cap SMEM at 0xC000. kBlockN=64 brings s_q+s_k+s_v+s_scores+s_phys/s_off
  // down to ~41KB.
  constexpr int kBlockN = 64;
  constexpr int kDimsPerThread = kHeadDim / kWarpSize;

  static_assert(kHeadDim % kWarpSize == 0, "head_dim must be divisible by 32.");

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  if (warp_id >= kWarps) {
    return;
  }

  const int head_idx = static_cast<int>(blockIdx.x);
  const int seqidx = static_cast<int>(blockIdx.y);
  const int m_block = static_cast<int>(blockIdx.z);

  const TIndex qstart = cu_seqlens_q[seqidx];
  const TIndex qend = cu_seqlens_q[seqidx + 1];
  const int qlen = static_cast<int>(qend - qstart);
  if (qlen <= 0) {
    return;
  }

  const int m_start = m_block * kBlockM;
  // Uniform early return for empty tail tiles (avoid deadlock with
  // __syncthreads()).
  if (m_start >= qlen) {
    return;
  }

  const int kv_len_total = static_cast<int>(total_kv_lens[seqidx]);
  const int history_len = kv_len_total - qlen;

  // Clamp max k length for this CTA based on the last active query row in the
  // tile.
  const int max_qin_tile = min(m_start + kBlockM, qlen);
  const int max_allowed_k_len = min(history_len + max_qin_tile, kv_len_total);

  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;
  const float alibi_slope_log2 = alibi_slope * kLog2e;

  const int pbs = static_cast<int>(page_block_size);

  const TIndex* block_table =
      block_tables + static_cast<int64_t>(seqidx) *
                         static_cast<int64_t>(block_table_batch_stride);

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
    const int qtoken_local = m_start + r;
    if (qtoken_local < qlen) {
      const int64_t qtoken = qstart + static_cast<int64_t>(qtoken_local);
      const half* qptr =
          q + qtoken * qstride + static_cast<int64_t>(head_idx) * qhead_stride;
      s_q[r * kHeadDimSmem + d] = qptr[d];
    } else {
      s_q[r * kHeadDimSmem + d] = __float2half_rn(0.0f);
    }
  }
  __syncthreads();

  // Two rows per warp: row0=warp_id, row1=warp_id+kWarps.
  const int row0 = warp_id;
  const int row1 = warp_id + kWarps;
  const bool active0 = (row0 < kBlockM) && ((m_start + row0) < qlen);
  const bool active1 = (row1 < kBlockM) && ((m_start + row1) < qlen);
  const int allowed0 =
      active0 ? min(history_len + (m_start + row0) + 1, kv_len_total) : 0;
  const int allowed1 =
      active1 ? min(history_len + (m_start + row1) + 1, kv_len_total) : 0;

  float m0 = -INFINITY, l0 = 0.0f;
  float m1 = -INFINITY, l1 = 0.0f;
  float acc0[kDimsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc1[kDimsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Iterate over K/V tiles.
  for (int k_base = 0; k_base < max_allowed_k_len; k_base += kBlockN) {
    // Map logical k positions to physical blocks for this tile (pad the tail
    // with -1).
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
        const half* k_ptr = k_cache +
                            static_cast<int64_t>(phys) * k_batch_stride +
                            static_cast<int64_t>(off) * k_row_stride +
                            static_cast<int64_t>(kv_head_idx) * k_head_stride;
        const half* v_ptr = v_cache +
                            static_cast<int64_t>(phys) * v_batch_stride +
                            static_cast<int64_t>(off) * v_row_stride +
                            static_cast<int64_t>(kv_head_idx) * v_head_stride;
        s_k[t * kHeadDimSmem + d] = k_ptr[d];
        s_v[t * kHeadDimSmem + d] = v_ptr[d];
      } else {
        s_k[t * kHeadDimSmem + d] = __float2half_rn(0.0f);
        s_v[t * kHeadDimSmem + d] = __float2half_rn(0.0f);
      }
    }
    __syncthreads();

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    // WMMA: each warp computes scores for 16 keys (one 16-column slice of the K
    // tile) across all 16 rows. For kBlockN=64, only the first 4 warps
    // participate in WMMA score computation.
    namespace wmma = nvcuda::wmma;
    constexpr int kNSub = kBlockN / 16;
    if (warp_id < kNSub) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);

      const int n_sub = warp_id;  // [0, kNSub)
      const half* qtile = s_q;
      const half* k_tile = s_k + (n_sub * 16) * kHeadDimSmem;
      // K loop (head_dim=128).
#pragma unroll
      for (int kk = 0; kk < (kHeadDim / 16); ++kk) {
        wmma::load_matrix_sync(a_frag, qtile + kk * 16, kHeadDimSmem);
        wmma::load_matrix_sync(b_frag, k_tile + kk * 16, kHeadDimSmem);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      float* scores_tile = s_scores + n_sub * 16;
      wmma::store_matrix_sync(scores_tile, c_frag, kBlockN,
                              wmma::mem_row_major);
    }
#else
    // No WMMA support on this architecture: fall back to scalar dot in the
    // existing kernels. (We keep scores as 0 so this kernel is effectively
    // incorrect; host dispatch must avoid selecting it.)
    if (threadIdx.x == 0) {
      // Intentionally empty.
    }
#endif
    __syncthreads();

    // Online softmax + V update per row handled by the same warp across tiles.
    if (row0 < kBlockM) {
      PagedAttentionPrefillMmaScoreUpdateRow<kWarpSize, kBlockN, kHeadDim,
                                             kDimsPerThread>(
          lane, k_base, allowed0, s_scores + row0 * kBlockN, s_v, scale_log2,
          alibi_slope_log2, m0, l0, acc0);
    }
    if (row1 < kBlockM) {
      PagedAttentionPrefillMmaScoreUpdateRow<kWarpSize, kBlockN, kHeadDim,
                                             kDimsPerThread>(
          lane, k_base, allowed1, s_scores + row1 * kBlockN, s_v, scale_log2,
          alibi_slope_log2, m1, l1, acc1);
    }
    __syncthreads();
  }

  // Write outputs.
  if (row0 < kBlockM) {
    PagedAttentionPrefillMmaScoreWriteRow<TIndex, kWarpSize, kHeadDim,
                                          kDimsPerThread>(
        lane, active0, m_start + row0, qstart, head_idx, out, o_stride,
        o_head_stride, l0, acc0);
  }
  if (row1 < kBlockM) {
    PagedAttentionPrefillMmaScoreWriteRow<TIndex, kWarpSize, kHeadDim,
                                          kDimsPerThread>(
        lane, active1, m_start + row1, qstart, head_idx, out, o_stride,
        o_head_stride, l1, acc1);
  }
}

#endif  // !defined(ENABLE_HYGON_API)

}  // namespace op::paged_attention_prefill::cuda

namespace infini::ops {

template <typename TIndex>
__device__ __forceinline__ std::size_t PagedPrefillFindSeqId(
    std::size_t token_idx, const TIndex* cum_seqlens_q, std::size_t num_seqs) {
  std::size_t low = 0;
  std::size_t high = num_seqs;

  while (low < high) {
    std::size_t mid = (low + high) >> 1;
    std::size_t begin = static_cast<std::size_t>(cum_seqlens_q[mid]);
    std::size_t end = static_cast<std::size_t>(cum_seqlens_q[mid + 1]);

    if (token_idx >= begin && token_idx < end) {
      return mid;
    }

    if (token_idx < begin) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }

  return 0;
}

template <Device::Type kDev, typename TIndex, typename TData, typename TLengths>
__global__ void PagedAttentionPrefillInfinilmHd128WarpCta8PipeKernel(
    TData* __restrict__ out, const TData* __restrict__ q,
    const TData* __restrict__ k_cache, const TData* __restrict__ v_cache,
    const TIndex* __restrict__ block_tables, TLengths seqlens,
    const TIndex* __restrict__ cum_seqlens_q,
    const float* __restrict__ alibi_slopes, std::size_t num_kv_heads,
    float scale, std::size_t max_num_blocks_per_seq, std::size_t block_size,
    std::ptrdiff_t block_table_batch_stride, std::ptrdiff_t qstride,
    std::ptrdiff_t qhead_stride, std::ptrdiff_t k_cacheblock_stride,
    std::ptrdiff_t k_cacheslot_stride, std::ptrdiff_t k_cachehead_stride,
    std::ptrdiff_t v_cacheblock_stride, std::ptrdiff_t v_cacheslot_stride,
    std::ptrdiff_t v_cachehead_stride, std::ptrdiff_t outstride,
    std::ptrdiff_t outhead_stride) {
  op::paged_attention_prefill::cuda::
      PagedAttentionPrefillWarpCtaKernelPipelined<kDev, TIndex, TData, 128, 8,
                                                  32, 2>(
          out, q, k_cache, v_cache, block_tables, seqlens, cum_seqlens_q,
          alibi_slopes, num_kv_heads, scale, max_num_blocks_per_seq, block_size,
          block_table_batch_stride, qstride, qhead_stride, k_cacheblock_stride,
          k_cacheslot_stride, k_cachehead_stride, v_cacheblock_stride,
          v_cacheslot_stride, v_cachehead_stride, outstride, outhead_stride);
}

template <typename TData, typename TIndex, int kHeadSize>
__global__ void PagedAttentionPrefillInfinilmKernel(
    TData* __restrict__ out, const TData* __restrict__ q,
    const TData* __restrict__ k_cache, const TData* __restrict__ v_cache,
    const TIndex* __restrict__ block_tables, const TIndex* __restrict__ seqlens,
    const TIndex* __restrict__ cum_seqlens_q,
    const float* __restrict__ alibi_slopes, std::size_t num_heads,
    std::size_t num_kv_heads, float scale, std::size_t max_num_blocks_per_seq,
    std::size_t block_size, std::ptrdiff_t k_cacheblock_stride,
    std::ptrdiff_t k_cachehead_stride, std::ptrdiff_t k_cacheslot_stride,
    std::ptrdiff_t v_cacheblock_stride, std::ptrdiff_t v_cachehead_stride,
    std::ptrdiff_t v_cacheslot_stride, std::ptrdiff_t qstride,
    std::ptrdiff_t qhead_stride, std::ptrdiff_t outstride,
    std::ptrdiff_t outhead_stride, std::ptrdiff_t block_table_batch_stride,
    std::size_t num_seqs) {
  constexpr int kWarpSize = 32;
  static_assert(kHeadSize == 64 || kHeadSize == 128,
                "PagedAttentionPrefillInfinilm supports head sizes 64 and 128");
  static_assert(kHeadSize % kWarpSize == 0,
                "head size must be divisible by 32");

  const std::size_t global_token_idx = blockIdx.x;
  const std::size_t head_idx = blockIdx.y;
  const int lane = threadIdx.x;
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;
  constexpr float kLog2e = 1.4426950408889634f;

  __shared__ float reduce_buf[kWarpSize];
  __shared__ float state_buf[2];

  const std::size_t seqidx =
      PagedPrefillFindSeqId(global_token_idx, cum_seqlens_q, num_seqs);
  const std::size_t qbegin = static_cast<std::size_t>(cum_seqlens_q[seqidx]);
  const std::size_t qend = static_cast<std::size_t>(cum_seqlens_q[seqidx + 1]);
  const int qlen = static_cast<int>(qend - qbegin);
  const int qtoken_local = static_cast<int>(global_token_idx - qbegin);
  if (qtoken_local < 0 || qtoken_local >= qlen) {
    return;
  }

  const int total_kv_len = static_cast<int>(seqlens[seqidx]);
  const int history_len = total_kv_len - qlen;
  const int allowed_k_len = history_len + qtoken_local + 1;
  if (allowed_k_len <= 0) {
    return;
  }

  const int queries_per_kv = static_cast<int>(num_heads / num_kv_heads);
  const int kv_head_idx = static_cast<int>(head_idx) / queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.0f : alibi_slopes[head_idx];
  const float scale_log2 = scale * kLog2e;
  const TIndex* block_table = block_tables + seqidx * block_table_batch_stride;

  const TData* qptr = q + global_token_idx * qstride + head_idx * qhead_stride;
  TData* outptr =
      out + global_token_idx * outstride + head_idx * outhead_stride;

  float qreg[kDimsPerThread];
  float acc[kDimsPerThread];
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    qreg[i] = static_cast<float>(qptr[dim]);
    acc[i] = 0.0f;
  }

  float m = -FLT_MAX;
  float l = 0.0f;
  const int page_block_size = static_cast<int>(block_size);
  int t_base = 0;
  for (int logical_block = 0;
       t_base < allowed_k_len &&
       logical_block < static_cast<int>(max_num_blocks_per_seq);
       ++logical_block, t_base += page_block_size) {
    const int physical_block = static_cast<int>(block_table[logical_block]);
    const TData* k_base = k_cache + physical_block * k_cacheblock_stride +
                          kv_head_idx * k_cachehead_stride;
    const TData* v_base = v_cache + physical_block * v_cacheblock_stride +
                          kv_head_idx * v_cachehead_stride;
    const int token_end = min(page_block_size, allowed_k_len - t_base);

    for (int token_in_block = 0; token_in_block < token_end; ++token_in_block) {
      const int token_idx = t_base + token_in_block;
      const TData* k_ptr = k_base + token_in_block * k_cacheslot_stride;
      const TData* v_ptr = v_base + token_in_block * v_cacheslot_stride;

      float qk = 0.0f;
#pragma unroll
      for (int i = 0; i < kDimsPerThread; ++i) {
        const int dim = lane * kDimsPerThread + i;
        qk += qreg[i] * static_cast<float>(k_ptr[dim]);
      }

      reduce_buf[lane] = qk;
      __syncthreads();
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
          reduce_buf[lane] += reduce_buf[lane + offset];
        }
        __syncthreads();
      }

      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0) {
        float score = reduce_buf[0] * scale_log2;
        if (alibi_slope != 0.0f) {
          score += (alibi_slope *
                    static_cast<float>(token_idx - (allowed_k_len - 1))) *
                   kLog2e;
        }
        const float m_new = fmaxf(m, score);
        alpha = exp2f(m - m_new);
        beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;
        state_buf[0] = alpha;
        state_buf[1] = beta;
      }
      __syncthreads();
      alpha = state_buf[0];
      beta = state_buf[1];

#pragma unroll
      for (int i = 0; i < kDimsPerThread; ++i) {
        const int dim = lane * kDimsPerThread + i;
        acc[i] = acc[i] * alpha + beta * static_cast<float>(v_ptr[dim]);
      }
    }
  }

  if (lane == 0) {
    state_buf[0] = 1.0f / (l + 1e-6f);
  }
  __syncthreads();
  const float inv_l = state_buf[0];

#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    outptr[dim] = static_cast<TData>(acc[i] * inv_l);
  }
}

}  // namespace infini::ops

#endif
