#ifndef INFINI_OPS_CUDA_PAGED_ATTENTION_INFINILM_KERNEL_CUH_
#define INFINI_OPS_CUDA_PAGED_ATTENTION_INFINILM_KERNEL_CUH_

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "native/cuda/caster.cuh"

namespace op::paged_attention::cuda {

// Standalone InfiniOps defines WITH_ILUVATAR; the legacy InfiniCore xmake
// integration defines ENABLE_ILUVATAR_API. Keep both build paths on the
// Iluvatar-safe warp and shared-memory implementation.
#if defined(WITH_ILUVATAR) || defined(ENABLE_ILUVATAR_API)
#define INFINI_OPS_PAGED_ATTENTION_ILUVATAR 1
#endif

struct OnlineSoftmaxState {
  float m = -INFINITY;

  float l = 0.0f;

  __device__ __forceinline__ void Update(float x, float& alpha, float& beta) {
    const float m_new = fmaxf(m, x);
    alpha = expf(m - m_new);
    beta = expf(x - m_new);
    l = l * alpha + beta;
    m = m_new;
  }
};

__device__ __forceinline__ float WarpReduceSum(float x) {
#if defined(INFINI_OPS_PAGED_ATTENTION_ILUVATAR)
  // Iluvatar may use warp size 64; __shfl_sync(0xffffffff) only covers 32
  // threads. Use shared-memory tree reduce for portability across warp sizes.
  constexpr int kMaxWarps = 16;
  __shared__ float _reduce_buf[kMaxWarps * 32];
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x / 32;
  _reduce_buf[threadIdx.x] = x;
  __syncthreads();
  for (int offset = 16; offset > 0; offset >>= 1) {
    if (lane < offset) {
      _reduce_buf[warp_id * 32 + lane] +=
          _reduce_buf[warp_id * 32 + lane + offset];
    }
    __syncthreads();
  }
  return _reduce_buf[warp_id * 32];
#else
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
  return x;
#endif
}

__device__ __forceinline__ float WarpBroadcast(float x, int src_lane) {
#if defined(INFINI_OPS_PAGED_ATTENTION_ILUVATAR)
  __shared__ float _bcast_buf[16];
  const int warp_id = threadIdx.x / 32;
  if ((threadIdx.x & 31) == src_lane) {
    _bcast_buf[warp_id] = x;
  }
  __syncthreads();
  return _bcast_buf[warp_id];
#else
  return __shfl_sync(0xffffffff, x, src_lane);
#endif
}

__device__ __forceinline__ float WarpReduceMax(float x) {
#if defined(INFINI_OPS_PAGED_ATTENTION_ILUVATAR)
  __shared__ float _reduce_buf[16 * 32];
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x / 32;
  _reduce_buf[threadIdx.x] = x;
  __syncthreads();
  for (int offset = 16; offset > 0; offset >>= 1) {
    if (lane < offset) {
      float other = _reduce_buf[warp_id * 32 + lane + offset];
      float cur = _reduce_buf[warp_id * 32 + lane];
      _reduce_buf[warp_id * 32 + lane] = fmaxf(cur, other);
    }
    __syncthreads();
  }
  return _reduce_buf[warp_id * 32];
#else
  for (int offset = 16; offset > 0; offset >>= 1) {
    x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
  }
  return x;
#endif
}

__device__ __forceinline__ unsigned int CvtaToShared(const void* ptr) {
#if defined(INFINI_OPS_PAGED_ATTENTION_ILUVATAR)
  return static_cast<unsigned int>(reinterpret_cast<uintptr_t>(ptr));
#else
  return static_cast<unsigned int>(__cvta_generic_to_shared(ptr));
#endif
}

__device__ __forceinline__ void CpAsyncCaSharedGlobal16(
    void* dst_shared, const void* src_global) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const unsigned int dst = CvtaToShared(dst_shared);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(dst),
               "l"(src_global));
#else
  auto* dst = reinterpret_cast<uint4*>(dst_shared);
  const auto* src = reinterpret_cast<const uint4*>(src_global);
  *dst = *src;
#endif
}

__device__ __forceinline__ void CpAsyncCommit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int kWaitGroup>
__device__ __forceinline__ void CpAsyncWaitGroup() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kWaitGroup));
#endif
}

// cp.async.wait_group requires a compile-time immediate, so for small fixed
// stage counts we provide a tiny runtime switch.
__device__ __forceinline__ void CpAsyncWaitGroupRt(int n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  if (n <= 0) {
    CpAsyncWaitGroup<0>();
  } else if (n == 1) {
    CpAsyncWaitGroup<1>();
  } else {
    // Clamp to 2 because v0.4 CTA kernel uses kStages=3.
    CpAsyncWaitGroup<2>();
  }
#else
  (void)n;
#endif
}

__device__ __forceinline__ void CpAsyncWaitAll() { CpAsyncWaitGroup<0>(); }

template <infini::ops::Device::Type kDev, typename TIndex, typename TData,
          int kHeadSize>
__device__ void FlashAttentionDecodeWarpKernel(
    TData* out, const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, const TIndex* cache_lens,
    const float* alibi_slopes, size_t num_kv_heads, float scale,
    size_t max_num_blocks_per_seq, size_t page_block_size, ptrdiff_t qstride,
    ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {
  const int seqidx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int lane = threadIdx.x;
  constexpr int kWarpSize = 32;
  static_assert(kHeadSize == 64 || kHeadSize == 128 || kHeadSize == 192 ||
                    kHeadSize == 576,
                "Only head_size 64/128/192/576 supported in v0.4.");
  static_assert(kHeadSize % kWarpSize == 0,
                "kHeadSize must be divisible by 32.");
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;

  const int seqlen = static_cast<int>(cache_lens[seqidx]);
  if (seqlen <= 0) {
    return;
  }

  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  const TIndex* block_table =
      block_tables + seqidx * static_cast<int>(max_num_blocks_per_seq);

  // q/out are [num_seqs, num_heads, head_size]
  const TData* qptr = q + seqidx * qstride + head_idx * kHeadSize;
  TData* outptr = out + seqidx * o_stride + head_idx * kHeadSize;

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

  // Iterate by blocks to avoid per-token division/mod and redundant block_table
  // loads. Note: Per-token cp.async prefetching is generally too fine-grained
  // for decode and can regress. We keep the warp kernel simple and reserve
  // cp.async pipelining for CTA tile kernels.
  int t_base = 0;
  for (int logical_block = 0; t_base < seqlen; ++logical_block, t_base += pbs) {
    int physical_block = 0;
    if (lane == 0) {
      physical_block = static_cast<int>(block_table[logical_block]);
    }
    physical_block = __shfl_sync(0xffffffff, physical_block, 0);

    const TData* k_base =
        k_cache + physical_block * k_batch_stride + kv_head_idx * k_head_stride;
    const TData* v_base =
        v_cache + physical_block * v_batch_stride + kv_head_idx * v_head_stride;

    const int token_end = min(pbs, seqlen - t_base);
    for (int token_in_block = 0; token_in_block < token_end; ++token_in_block) {
      const int t = t_base + token_in_block;
      const TData* k_ptr = k_base + token_in_block * k_row_stride;
      const TData* v_ptr = v_base + token_in_block * v_row_stride;

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

      qk = WarpReduceSum(qk);

      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0) {
        float score = qk * scale_log2;
        if (alibi_slope != 0.0f) {
          score +=
              (alibi_slope * static_cast<float>(t - (seqlen - 1))) * kLog2e;
        }
        const float m_new = fmaxf(m, score);
        alpha = exp2f(m - m_new);
        beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;
      }

      alpha = __shfl_sync(0xffffffff, alpha, 0);
      beta = __shfl_sync(0xffffffff, beta, 0);

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
  inv_l = __shfl_sync(0xffffffff, inv_l, 0);

#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    const float o = acc[i] * inv_l;
    outptr[dim] = infini::ops::Caster<kDev>::template Cast<TData>(o);
  }
}

// Split-KV decode (FA2-style): each split scans a shard of KV and writes
// partial (m, l, acc) to workspace, then a combine kernel merges splits into
// final out.
template <typename TIndex, typename TData, int kHeadSize>
__device__ void FlashAttentionDecodeSplitKvWarpKernel(
    float* partial_acc,  // [num_splits, num_seqs, num_heads, head_size]
    float* partial_m,    // [num_splits, num_seqs, num_heads]
    float* partial_l,    // [num_splits, num_seqs, num_heads]
    const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, const TIndex* cache_lens,
    const float* alibi_slopes, size_t num_kv_heads, float scale,
    size_t max_num_blocks_per_seq, size_t page_block_size, ptrdiff_t qstride,
    ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride,
    int num_splits) {
  const int seqidx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int split_idx = static_cast<int>(blockIdx.z);
  const int lane = threadIdx.x;
  constexpr int kWarpSize = 32;
  static_assert(kHeadSize == 64 || kHeadSize == 128 || kHeadSize == 192 ||
                    kHeadSize == 576,
                "Only head_size 64/128/192/576 supported in v0.4.");
  static_assert(kHeadSize % kWarpSize == 0,
                "kHeadSize must be divisible by 32.");
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;

  const int pbs = static_cast<int>(page_block_size);
  const int raw_seqlen = static_cast<int>(cache_lens[seqidx]);
  const int max_reachable_tokens =
      static_cast<int>(max_num_blocks_per_seq) * pbs;
  const int seqlen = min(raw_seqlen, max_reachable_tokens);
  if (seqlen <= 0 || num_splits <= 0) {
    return;
  }

  // Split the accessible [0, seqlen) range into num_splits contiguous shards.
  const int shard = (seqlen + num_splits - 1) / num_splits;
  const int start = split_idx * shard;
  const int end = min(seqlen, start + shard);
  if (start >= end) {
    // Empty shard => write neutral element.
    const int n = gridDim.y * gridDim.x;
    const int idx = (split_idx * n + seqidx * gridDim.x + head_idx);
    if (lane == 0) {
      partial_m[idx] = -INFINITY;
      partial_l[idx] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
      const int dim = lane * kDimsPerThread + i;
      partial_acc[idx * kHeadSize + dim] = 0.0f;
    }
    return;
  }

  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  const TIndex* block_table =
      block_tables + seqidx * static_cast<int>(max_num_blocks_per_seq);
  const TData* qptr = q + seqidx * qstride + head_idx * kHeadSize;

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

  // Scan only [start, end).
  int t = start;
  int logical_block = t / pbs;
  int token_in_block = t - logical_block * pbs;
  for (; t < end; ++logical_block) {
    int physical_block = 0;
    if (lane == 0) {
      physical_block = static_cast<int>(block_table[logical_block]);
    }
    physical_block = __shfl_sync(0xffffffff, physical_block, 0);

    const TData* k_base =
        k_cache + physical_block * k_batch_stride + kv_head_idx * k_head_stride;
    const TData* v_base =
        v_cache + physical_block * v_batch_stride + kv_head_idx * v_head_stride;

    const int token_end = min(pbs, end - logical_block * pbs);
    for (; token_in_block < token_end && t < end; ++token_in_block, ++t) {
      const TData* k_ptr = k_base + token_in_block * k_row_stride;
      const TData* v_ptr = v_base + token_in_block * v_row_stride;

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

      qk = WarpReduceSum(qk);

      float alpha = 1.0f;
      float beta = 0.0f;
      if (lane == 0) {
        float score = qk * scale_log2;
        if (alibi_slope != 0.0f) {
          score +=
              (alibi_slope * static_cast<float>(t - (seqlen - 1))) * kLog2e;
        }
        const float m_new = fmaxf(m, score);
        alpha = exp2f(m - m_new);
        beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;
      }

      alpha = __shfl_sync(0xffffffff, alpha, 0);
      beta = __shfl_sync(0xffffffff, beta, 0);

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
    token_in_block = 0;
  }

  const int n = gridDim.y * gridDim.x;
  const int idx = (split_idx * n + seqidx * gridDim.x + head_idx);
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

template <infini::ops::Device::Type kDev, typename TData, int kHeadSize>
__device__ void FlashAttentionDecodeSplitKvCombineWarpKernel(
    TData* out,
    const float* partial_acc,  // [num_splits, num_seqs, num_heads, head_size]
    const float* partial_m,    // [num_splits, num_seqs, num_heads]
    const float* partial_l,    // [num_splits, num_seqs, num_heads]
    int num_splits, ptrdiff_t o_stride) {
  const int seqidx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int lane = threadIdx.x;
  constexpr int kWarpSize = 32;
  static_assert(kHeadSize % kWarpSize == 0,
                "kHeadSize must be divisible by 32.");
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;

  const int n = gridDim.y * gridDim.x;
  const int base = (seqidx * gridDim.x + head_idx);

  float m = -INFINITY;
  if (lane == 0) {
    for (int s = 0; s < num_splits; ++s) {
      m = fmaxf(m, partial_m[s * n + base]);
    }
  }
  m = __shfl_sync(0xffffffff, m, 0);

  float l = 0.0f;
  if (lane == 0) {
    for (int s = 0; s < num_splits; ++s) {
      const float ms = partial_m[s * n + base];
      const float ls = partial_l[s * n + base];
      if (ls > 0.0f) {
        l += ls * exp2f(ms - m);
      }
    }
  }
  l = __shfl_sync(0xffffffff, l, 0);
  const float inv_l = 1.0f / (l + 1e-6f);

  // Combine acc for each dim.
  TData* outptr = out + seqidx * o_stride + head_idx * kHeadSize;
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int dim = lane * kDimsPerThread + i;
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      const float ms = partial_m[s * n + base];
      const float w = exp2f(ms - m);
      acc += partial_acc[(s * n + base) * kHeadSize + dim] * w;
    }
    const float o = acc * inv_l;
    outptr[dim] = infini::ops::Caster<kDev>::template Cast<TData>(o);
  }
}

// Split-KV decode with a CTA tile kernel (FA2-style): each CTA scans a shard of
// KV, writes partial (m, l, acc) to workspace, then a combine kernel merges
// splits.
template <typename TIndex, typename TData, int kHeadSize, int kCtaThreads,
          int kTokensPerTile>
__device__ void FlashAttentionDecodeSplitKvCtaKernel(
    float* partial_acc,  // [num_splits, num_seqs, num_heads, head_size]
    float* partial_m,    // [num_splits, num_seqs, num_heads]
    float* partial_l,    // [num_splits, num_seqs, num_heads]
    const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, const TIndex* cache_lens,
    const float* alibi_slopes, size_t num_kv_heads, float scale,
    size_t max_num_blocks_per_seq, size_t page_block_size, ptrdiff_t qstride,
    ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride,
    int num_splits) {
  constexpr int kWarpSize = 32;
  static_assert(kCtaThreads % kWarpSize == 0,
                "kCtaThreads must be a multiple of 32.");
  static_assert(kTokensPerTile > 0 && kTokensPerTile <= 16,
                "kTokensPerTile should stay small.");
  constexpr int NUM_WARPS = kCtaThreads / kWarpSize;

  static_assert(kHeadSize == 64 || kHeadSize == 128 || kHeadSize == 192 ||
                    kHeadSize == 576,
                "Only head_size 64/128/192/576 supported in v0.4.");
  static_assert(kHeadSize % kCtaThreads == 0,
                "kHeadSize must be divisible by kCtaThreads.");
  constexpr int kPack =
      kHeadSize / kCtaThreads;  // 2 (64@32t, 128@64t) or 4 (128@32t)
  static_assert(kPack == 2 || kPack == 4,
                "v0.4 split-kv CTA kernel supports kPack=2/4 only.");
  constexpr int kPackedDims = kCtaThreads;
  constexpr int kComputeWarps = (kPackedDims + kWarpSize - 1) / kWarpSize;

  const int seqidx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int split_idx = static_cast<int>(blockIdx.z);
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int warp_id = tid / kWarpSize;

  const int pbs = static_cast<int>(page_block_size);
  const int raw_seqlen = static_cast<int>(cache_lens[seqidx]);
  const int max_reachable_tokens =
      static_cast<int>(max_num_blocks_per_seq) * pbs;
  const int seqlen = min(raw_seqlen, max_reachable_tokens);
  if (seqlen <= 0 || num_splits <= 0) {
    return;
  }

  // Split the accessible [0, seqlen) range into num_splits contiguous shards.
  const int shard = (seqlen + num_splits - 1) / num_splits;
  const int start = split_idx * shard;
  const int end = min(seqlen, start + shard);

  const int n = gridDim.y * gridDim.x;
  const int idx = (split_idx * n + seqidx * gridDim.x + head_idx);

  if (start >= end) {
    // Empty shard => write neutral element.
    if (tid == 0) {
      partial_m[idx] = -INFINITY;
      partial_l[idx] = 0.0f;
    }
    const int dim = tid * kPack;
    if constexpr (kPack == 2) {
      partial_acc[idx * kHeadSize + dim + 0] = 0.0f;
      partial_acc[idx * kHeadSize + dim + 1] = 0.0f;
    } else {
      partial_acc[idx * kHeadSize + dim + 0] = 0.0f;
      partial_acc[idx * kHeadSize + dim + 1] = 0.0f;
      partial_acc[idx * kHeadSize + dim + 2] = 0.0f;
      partial_acc[idx * kHeadSize + dim + 3] = 0.0f;
    }
    return;
  }

  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const TIndex* block_table =
      block_tables + seqidx * static_cast<int>(max_num_blocks_per_seq);
  const TData* qptr = q + seqidx * qstride + head_idx * kHeadSize;

  const int dim = tid * kPack;
  float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<TData, half>) {
    if constexpr (kPack == 2) {
      const half2 qh2 = *reinterpret_cast<const half2*>(qptr + dim);
      const float2 qf = __half22float2(qh2);
      q0 = qf.x;
      q1 = qf.y;
    } else {
      const half2 qh2_0 = *reinterpret_cast<const half2*>(qptr + dim + 0);
      const half2 qh2_1 = *reinterpret_cast<const half2*>(qptr + dim + 2);
      const float2 qf0 = __half22float2(qh2_0);
      const float2 qf1 = __half22float2(qh2_1);
      q0 = qf0.x;
      q1 = qf0.y;
      q2 = qf1.x;
      q3 = qf1.y;
    }
  } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
    if constexpr (kPack == 2) {
      const __nv_bfloat162 qb2 =
          *reinterpret_cast<const __nv_bfloat162*>(qptr + dim);
      const float2 qf = __bfloat1622float2(qb2);
      q0 = qf.x;
      q1 = qf.y;
    } else {
      const __nv_bfloat162 qb2_0 =
          *reinterpret_cast<const __nv_bfloat162*>(qptr + dim + 0);
      const __nv_bfloat162 qb2_1 =
          *reinterpret_cast<const __nv_bfloat162*>(qptr + dim + 2);
      const float2 qf0 = __bfloat1622float2(qb2_0);
      const float2 qf1 = __bfloat1622float2(qb2_1);
      q0 = qf0.x;
      q1 = qf0.y;
      q2 = qf1.x;
      q3 = qf1.y;
    }
  } else
#endif
  {
    q0 = static_cast<float>(qptr[dim + 0]);
    q1 = static_cast<float>(qptr[dim + 1]);
    if constexpr (kPack == 4) {
      q2 = static_cast<float>(qptr[dim + 2]);
      q3 = static_cast<float>(qptr[dim + 3]);
    }
  }

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  float m = -INFINITY;
  float l = 0.0f;

  __shared__ float warp_sums[kTokensPerTile][kComputeWarps];
  __shared__ float alpha_shared;
  __shared__ float weights_shared[kTokensPerTile];

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  static_assert(sizeof(TData) == 2, "CTA split-kv kernel assumes fp16/bf16.");
  constexpr int kChunkElems = 8;  // 8 * 2 bytes = 16 bytes.
  constexpr int CHUNKS = kHeadSize / kChunkElems;
  constexpr int LOADS_PER_TILE = CHUNKS * kTokensPerTile;

  constexpr int kStages = 3;
  __shared__ __align__(16) TData sh_k[kStages][kTokensPerTile][kHeadSize];
  __shared__ __align__(16) TData sh_v[kStages][kTokensPerTile][kHeadSize];

  const int first_block = start / pbs;
  const int last_block = (end - 1) / pbs;

  for (int logical_block = first_block; logical_block <= last_block;
       ++logical_block) {
    const int physical_block = static_cast<int>(block_table[logical_block]);
    const TData* k_base =
        k_cache + physical_block * k_batch_stride + kv_head_idx * k_head_stride;
    const TData* v_base =
        v_cache + physical_block * v_batch_stride + kv_head_idx * v_head_stride;

    const int t_base = logical_block * pbs;
    const int token_begin =
        (logical_block == first_block) ? (start - t_base) : 0;
    const int token_end = (logical_block == last_block) ? (end - t_base) : pbs;
    const int token_count = token_end - token_begin;
    if (token_count <= 0) {
      continue;
    }

    const int num_tiles = (token_count + kTokensPerTile - 1) / kTokensPerTile;
    int pending_groups = 0;
    const int preload = min(kStages, num_tiles);
    for (int ti = 0; ti < preload; ++ti) {
      const int token_in_block = token_begin + ti * kTokensPerTile;
      const int tile_n = min(kTokensPerTile, token_end - token_in_block);
      for (int li = tid; li < LOADS_PER_TILE; li += kCtaThreads) {
        const int tok = li / CHUNKS;
        const int chunk = li - tok * CHUNKS;
        const int off = chunk * kChunkElems;
        if (tok < tile_n) {
          const TData* k_src =
              k_base + (token_in_block + tok) * k_row_stride + off;
          const TData* v_src =
              v_base + (token_in_block + tok) * v_row_stride + off;
          CpAsyncCaSharedGlobal16(&sh_k[ti][tok][off], k_src);
          CpAsyncCaSharedGlobal16(&sh_v[ti][tok][off], v_src);
        } else {
          reinterpret_cast<uint4*>(&sh_k[ti][tok][off])[0] =
              make_uint4(0, 0, 0, 0);
          reinterpret_cast<uint4*>(&sh_v[ti][tok][off])[0] =
              make_uint4(0, 0, 0, 0);
        }
      }
      CpAsyncCommit();
      ++pending_groups;
    }

    int desired_pending = pending_groups - 1;
    if (desired_pending < 0) {
      desired_pending = 0;
    }
    if (desired_pending > (kStages - 1)) {
      desired_pending = (kStages - 1);
    }
    CpAsyncWaitGroupRt(desired_pending);
    pending_groups = desired_pending;
    if constexpr (NUM_WARPS == 1) {
      __syncwarp();
    } else {
      __syncthreads();
    }

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      const int buf = tile_idx % kStages;
      const int token_in_block = token_begin + tile_idx * kTokensPerTile;
      const int tile_n = min(kTokensPerTile, token_end - token_in_block);

      float partial[kTokensPerTile];
#pragma unroll
      for (int j = 0; j < kTokensPerTile; ++j) {
        if (j < tile_n) {
          float k0 = 0.0f, k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;
#if defined(__CUDA_ARCH__)
          if constexpr (std::is_same_v<TData, half>) {
            if constexpr (kPack == 2) {
              const half2 kh2 =
                  *reinterpret_cast<const half2*>(&sh_k[buf][j][dim]);
              const float2 kf = __half22float2(kh2);
              k0 = kf.x;
              k1 = kf.y;
            } else {
              const half2 kh2_0 =
                  *reinterpret_cast<const half2*>(&sh_k[buf][j][dim + 0]);
              const half2 kh2_1 =
                  *reinterpret_cast<const half2*>(&sh_k[buf][j][dim + 2]);
              const float2 kf0 = __half22float2(kh2_0);
              const float2 kf1 = __half22float2(kh2_1);
              k0 = kf0.x;
              k1 = kf0.y;
              k2 = kf1.x;
              k3 = kf1.y;
            }
          } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
            if constexpr (kPack == 2) {
              const __nv_bfloat162 kb2 =
                  *reinterpret_cast<const __nv_bfloat162*>(&sh_k[buf][j][dim]);
              const float2 kf = __bfloat1622float2(kb2);
              k0 = kf.x;
              k1 = kf.y;
            } else {
              const __nv_bfloat162 kb2_0 =
                  *reinterpret_cast<const __nv_bfloat162*>(
                      &sh_k[buf][j][dim + 0]);
              const __nv_bfloat162 kb2_1 =
                  *reinterpret_cast<const __nv_bfloat162*>(
                      &sh_k[buf][j][dim + 2]);
              const float2 kf0 = __bfloat1622float2(kb2_0);
              const float2 kf1 = __bfloat1622float2(kb2_1);
              k0 = kf0.x;
              k1 = kf0.y;
              k2 = kf1.x;
              k3 = kf1.y;
            }
          } else
#endif
          {
            k0 = static_cast<float>(sh_k[buf][j][dim + 0]);
            k1 = static_cast<float>(sh_k[buf][j][dim + 1]);
            if constexpr (kPack == 4) {
              k2 = static_cast<float>(sh_k[buf][j][dim + 2]);
              k3 = static_cast<float>(sh_k[buf][j][dim + 3]);
            }
          }
          if constexpr (kPack == 2) {
            partial[j] = fmaf(q0, k0, q1 * k1);
          } else {
            partial[j] = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, q3 * k3)));
          }
        } else {
          partial[j] = 0.0f;
        }
      }

#pragma unroll
      for (int j = 0; j < kTokensPerTile; ++j) {
        const float sum = WarpReduceSum(partial[j]);
        if (lane == 0 && warp_id < kComputeWarps) {
          warp_sums[j][warp_id] = sum;
        }
      }

      if constexpr (NUM_WARPS == 1) {
        __syncwarp();
      } else {
        __syncthreads();
      }

      if (warp_id == 0) {
        float score = -INFINITY;
        if (lane < kTokensPerTile && lane < tile_n) {
          float qk = 0.0f;
#pragma unroll
          for (int w = 0; w < kComputeWarps; ++w) {
            qk += warp_sums[lane][w];
          }
          const int t = t_base + token_in_block + lane;
          score = qk * scale_log2;
          if (alibi_slope != 0.0f) {
            score +=
                (alibi_slope * static_cast<float>(t - (seqlen - 1))) * kLog2e;
          }
        }

        float tile_max = WarpReduceMax(score);
        tile_max = __shfl_sync(0xffffffff, tile_max, 0);

        float m_new = 0.0f;
        if (lane == 0) {
          m_new = fmaxf(m, tile_max);
        }
        m_new = __shfl_sync(0xffffffff, m_new, 0);

        float w = 0.0f;
        if (lane < kTokensPerTile && lane < tile_n) {
          w = exp2f(score - m_new);
        }
        if (lane < kTokensPerTile) {
          weights_shared[lane] = (lane < tile_n) ? w : 0.0f;
        }

        const float tile_sum = WarpReduceSum(w);
        if (lane == 0) {
          const float alpha = exp2f(m - m_new);
          alpha_shared = alpha;
          l = l * alpha + tile_sum;
          m = m_new;
        }
      }

      if constexpr (NUM_WARPS == 1) {
        __syncwarp();
      } else {
        __syncthreads();
      }

      const float alpha = alpha_shared;
      float sum_wv0 = 0.0f, sum_wv1 = 0.0f, sum_wv2 = 0.0f, sum_wv3 = 0.0f;
#pragma unroll
      for (int j = 0; j < kTokensPerTile; ++j) {
        const float w = weights_shared[j];
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
#if defined(__CUDA_ARCH__)
        if constexpr (std::is_same_v<TData, half>) {
          if constexpr (kPack == 2) {
            const half2 vh2 =
                *reinterpret_cast<const half2*>(&sh_v[buf][j][dim]);
            const float2 vf = __half22float2(vh2);
            v0 = vf.x;
            v1 = vf.y;
          } else {
            const half2 vh2_0 =
                *reinterpret_cast<const half2*>(&sh_v[buf][j][dim + 0]);
            const half2 vh2_1 =
                *reinterpret_cast<const half2*>(&sh_v[buf][j][dim + 2]);
            const float2 vf0 = __half22float2(vh2_0);
            const float2 vf1 = __half22float2(vh2_1);
            v0 = vf0.x;
            v1 = vf0.y;
            v2 = vf1.x;
            v3 = vf1.y;
          }
        } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
          if constexpr (kPack == 2) {
            const __nv_bfloat162 vb2 =
                *reinterpret_cast<const __nv_bfloat162*>(&sh_v[buf][j][dim]);
            const float2 vf = __bfloat1622float2(vb2);
            v0 = vf.x;
            v1 = vf.y;
          } else {
            const __nv_bfloat162 vb2_0 =
                *reinterpret_cast<const __nv_bfloat162*>(
                    &sh_v[buf][j][dim + 0]);
            const __nv_bfloat162 vb2_1 =
                *reinterpret_cast<const __nv_bfloat162*>(
                    &sh_v[buf][j][dim + 2]);
            const float2 vf0 = __bfloat1622float2(vb2_0);
            const float2 vf1 = __bfloat1622float2(vb2_1);
            v0 = vf0.x;
            v1 = vf0.y;
            v2 = vf1.x;
            v3 = vf1.y;
          }
        } else
#endif
        {
          v0 = static_cast<float>(sh_v[buf][j][dim + 0]);
          v1 = static_cast<float>(sh_v[buf][j][dim + 1]);
          if constexpr (kPack == 4) {
            v2 = static_cast<float>(sh_v[buf][j][dim + 2]);
            v3 = static_cast<float>(sh_v[buf][j][dim + 3]);
          }
        }
        sum_wv0 = fmaf(w, v0, sum_wv0);
        sum_wv1 = fmaf(w, v1, sum_wv1);
        if constexpr (kPack == 4) {
          sum_wv2 = fmaf(w, v2, sum_wv2);
          sum_wv3 = fmaf(w, v3, sum_wv3);
        }
      }
      acc0 = acc0 * alpha + sum_wv0;
      acc1 = acc1 * alpha + sum_wv1;
      if constexpr (kPack == 4) {
        acc2 = acc2 * alpha + sum_wv2;
        acc3 = acc3 * alpha + sum_wv3;
      }

      const int prefetch_tile = tile_idx + kStages;
      if (prefetch_tile < num_tiles) {
        const int token_prefetch = token_begin + prefetch_tile * kTokensPerTile;
        const int prefetch_n = min(kTokensPerTile, token_end - token_prefetch);
        for (int li = tid; li < LOADS_PER_TILE; li += kCtaThreads) {
          const int tok = li / CHUNKS;
          const int chunk = li - tok * CHUNKS;
          const int off = chunk * kChunkElems;
          if (tok < prefetch_n) {
            const TData* k_src =
                k_base + (token_prefetch + tok) * k_row_stride + off;
            const TData* v_src =
                v_base + (token_prefetch + tok) * v_row_stride + off;
            CpAsyncCaSharedGlobal16(&sh_k[buf][tok][off], k_src);
            CpAsyncCaSharedGlobal16(&sh_v[buf][tok][off], v_src);
          } else {
            reinterpret_cast<uint4*>(&sh_k[buf][tok][off])[0] =
                make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(&sh_v[buf][tok][off])[0] =
                make_uint4(0, 0, 0, 0);
          }
        }
        CpAsyncCommit();
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
        CpAsyncWaitGroupRt(desired_pending2);
        pending_groups = desired_pending2;
        if constexpr (NUM_WARPS == 1) {
          __syncwarp();
        } else {
          __syncthreads();
        }
      }
    }

    CpAsyncWaitAll();
    if constexpr (NUM_WARPS == 1) {
      __syncwarp();
    } else {
      __syncthreads();
    }
  }

  if (tid == 0) {
    partial_m[idx] = m;
    partial_l[idx] = l;
  }
  if constexpr (kPack == 2) {
    partial_acc[idx * kHeadSize + dim + 0] = acc0;
    partial_acc[idx * kHeadSize + dim + 1] = acc1;
  } else {
    partial_acc[idx * kHeadSize + dim + 0] = acc0;
    partial_acc[idx * kHeadSize + dim + 1] = acc1;
    partial_acc[idx * kHeadSize + dim + 2] = acc2;
    partial_acc[idx * kHeadSize + dim + 3] = acc3;
  }
}

template <typename TIndex, typename TData, int kHeadSize>
__device__ void FlashAttentionDecodeCtaPipelinedKernel(
    TData* out, const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, const TIndex* cache_lens,
    const float* alibi_slopes, size_t num_kv_heads, float scale,
    size_t max_num_blocks_per_seq, size_t page_block_size, ptrdiff_t qstride,
    ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {
  constexpr int kWarpSize = 32;
  static_assert(kHeadSize == 64 || kHeadSize == 128 || kHeadSize == 192 ||
                    kHeadSize == 576,
                "Only head_size 64/128/192/576 supported in v0.4.");
  static_assert(kHeadSize % kWarpSize == 0,
                "kHeadSize must be divisible by 32.");
  constexpr int NUM_WARPS = kHeadSize / kWarpSize;

  const int seqidx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int warp_id = tid / kWarpSize;

  const int seqlen = static_cast<int>(cache_lens[seqidx]);
  if (seqlen <= 0) {
    return;
  }

  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  const TIndex* block_table =
      block_tables + seqidx * static_cast<int>(max_num_blocks_per_seq);

  const TData* qptr = q + seqidx * qstride + head_idx * kHeadSize;
  TData* outptr = out + seqidx * o_stride + head_idx * kHeadSize;

  const float qval = static_cast<float>(qptr[tid]);
  float acc = 0.0f;

  float m = -INFINITY;
  float l = 0.0f;

  __shared__ TData sh_k[2][kHeadSize];
  __shared__ TData sh_v[2][kHeadSize];
  __shared__ float warp_sums[NUM_WARPS];
  __shared__ float alpha_s;
  __shared__ float beta_s;
  __shared__ int physical_block_s;
  constexpr int kChunkElems = 8;  // 8 * 2 bytes = 16 bytes.
  constexpr int CHUNKS = kHeadSize / kChunkElems;

  const int pbs = static_cast<int>(page_block_size);

  // Prefetch the very first token.
  int buf = 0;
  int t_base = 0;
  int token_in_block = 0;
  int logical_block = 0;
  {
    if (tid == 0) {
      physical_block_s = static_cast<int>(block_table[0]);
    }
    __syncthreads();
    const TData* k_base = k_cache + physical_block_s * k_batch_stride +
                          kv_head_idx * k_head_stride;
    const TData* v_base = v_cache + physical_block_s * v_batch_stride +
                          kv_head_idx * v_head_stride;
    if (tid < CHUNKS) {
      const int off = tid * kChunkElems;
      CpAsyncCaSharedGlobal16(&sh_k[buf][off],
                              (k_base + 0 * k_row_stride) + off);
      CpAsyncCaSharedGlobal16(&sh_v[buf][off],
                              (v_base + 0 * v_row_stride) + off);
    }
    CpAsyncCommit();
    CpAsyncWaitAll();
    __syncthreads();
  }

  for (int t = 0; t < seqlen; ++t) {
    // Compute current token location within paged KV.
    const int next_t = t + 1;
    const bool has_next = next_t < seqlen;

    if (has_next) {
      const int next_block = next_t / pbs;
      const int next_in_block = next_t - next_block * pbs;
      if (next_block != logical_block) {
        logical_block = next_block;
        if (tid == 0) {
          physical_block_s = static_cast<int>(block_table[logical_block]);
        }
        __syncthreads();
      }

      const TData* k_base = k_cache + physical_block_s * k_batch_stride +
                            kv_head_idx * k_head_stride;
      const TData* v_base = v_cache + physical_block_s * v_batch_stride +
                            kv_head_idx * v_head_stride;
      const TData* k_src = k_base + next_in_block * k_row_stride;
      const TData* v_src = v_base + next_in_block * v_row_stride;
      if (tid < CHUNKS) {
        const int off = tid * kChunkElems;
        CpAsyncCaSharedGlobal16(&sh_k[buf ^ 1][off], k_src + off);
        CpAsyncCaSharedGlobal16(&sh_v[buf ^ 1][off], v_src + off);
      }
      CpAsyncCommit();
    }

    // Dot: each thread handles one dim, reduce across head dim.
    const float k_val = static_cast<float>(sh_k[buf][tid]);
    float partial = qval * k_val;
    float warp_sum = WarpReduceSum(partial);
    if (lane == 0) {
      warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    float qk = 0.0f;
    if (warp_id == 0) {
      float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
      v = WarpReduceSum(v);
      if (lane == 0) {
        qk = v;
        float score = qk * scale_log2;
        if (alibi_slope != 0.0f) {
          score +=
              (alibi_slope * static_cast<float>(t - (seqlen - 1))) * kLog2e;
        }
        const float m_new = fmaxf(m, score);
        const float alpha = exp2f(m - m_new);
        const float beta = exp2f(score - m_new);
        l = l * alpha + beta;
        m = m_new;
        alpha_s = alpha;
        beta_s = beta;
      }
    }
    __syncthreads();

    const float alpha = alpha_s;
    const float beta = beta_s;
    const float v_val = static_cast<float>(sh_v[buf][tid]);
    acc = acc * alpha + beta * v_val;

    if (has_next) {
      CpAsyncWaitAll();
      __syncthreads();
      buf ^= 1;
    }
  }

  __shared__ float inv_l_s;
  if (tid == 0) {
    inv_l_s = 1.0f / (l + 1e-6f);
  }
  __syncthreads();
  outptr[tid] = static_cast<TData>(acc * inv_l_s);
}

template <typename TIndex, typename TData, int kHeadSize, int kCtaThreads,
          int kTokensPerTile>
__device__ void FlashAttentionDecodeCtaKernel(
    TData* out, const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, const TIndex* cache_lens,
    const float* alibi_slopes, size_t num_kv_heads, float scale,
    size_t max_num_blocks_per_seq, size_t page_block_size, ptrdiff_t qstride,
    ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {
  constexpr int kWarpSize = 32;
  static_assert(kCtaThreads % kWarpSize == 0,
                "kCtaThreads must be a multiple of 32.");
  static_assert(kTokensPerTile > 0 && kTokensPerTile <= 16,
                "kTokensPerTile should stay small.");
  constexpr int NUM_WARPS = kCtaThreads / kWarpSize;

  const int seqidx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int warp_id = tid / kWarpSize;

  // Each thread owns a small packed vector of head dims. This lets us shrink
  // the CTA to 1-2 warps and reduce block-wide synchronization overhead.
  static_assert(kHeadSize % kCtaThreads == 0,
                "kHeadSize must be divisible by kCtaThreads.");
  constexpr int kPack =
      kHeadSize / kCtaThreads;  // 2 (64@32t, 128@64t) or 4 (128@32t)
  static_assert(kPack == 2 || kPack == 4,
                "v0.4 CTA tile kernel supports kPack=2/4 only.");
  constexpr int kPackedDims = kCtaThreads;
  constexpr int kComputeWarps = (kPackedDims + kWarpSize - 1) / kWarpSize;
  const int dim = tid * kPack;

  const int seqlen = static_cast<int>(cache_lens[seqidx]);
  if (seqlen <= 0) {
    return;
  }

  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const TIndex* block_table =
      block_tables + seqidx * static_cast<int>(max_num_blocks_per_seq);

  // q/out are [num_seqs, num_heads, head_size]
  const TData* qptr = q + seqidx * qstride + head_idx * kHeadSize;
  TData* outptr = out + seqidx * o_stride + head_idx * kHeadSize;

  float q0 = 0.0f;
  float q1 = 0.0f;
  float q2 = 0.0f;
  float q3 = 0.0f;
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<TData, half>) {
    if constexpr (kPack == 2) {
      const half2 qh2 = *reinterpret_cast<const half2*>(qptr + dim);
      const float2 qf = __half22float2(qh2);
      q0 = qf.x;
      q1 = qf.y;
    } else {
      const half2 qh2_0 = *reinterpret_cast<const half2*>(qptr + dim + 0);
      const half2 qh2_1 = *reinterpret_cast<const half2*>(qptr + dim + 2);
      const float2 qf0 = __half22float2(qh2_0);
      const float2 qf1 = __half22float2(qh2_1);
      q0 = qf0.x;
      q1 = qf0.y;
      q2 = qf1.x;
      q3 = qf1.y;
    }
  } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
    if constexpr (kPack == 2) {
      const __nv_bfloat162 qb2 =
          *reinterpret_cast<const __nv_bfloat162*>(qptr + dim);
      const float2 qf = __bfloat1622float2(qb2);
      q0 = qf.x;
      q1 = qf.y;
    } else {
      const __nv_bfloat162 qb2_0 =
          *reinterpret_cast<const __nv_bfloat162*>(qptr + dim + 0);
      const __nv_bfloat162 qb2_1 =
          *reinterpret_cast<const __nv_bfloat162*>(qptr + dim + 2);
      const float2 qf0 = __bfloat1622float2(qb2_0);
      const float2 qf1 = __bfloat1622float2(qb2_1);
      q0 = qf0.x;
      q1 = qf0.y;
      q2 = qf1.x;
      q3 = qf1.y;
    }
  } else
#endif
  {
    q0 = static_cast<float>(qptr[dim + 0]);
    q1 = static_cast<float>(qptr[dim + 1]);
    if constexpr (kPack == 4) {
      q2 = static_cast<float>(qptr[dim + 2]);
      q3 = static_cast<float>(qptr[dim + 3]);
    }
  }

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  float m = -INFINITY;
  float l = 0.0f;

  // Only the compute warps contribute QK partial sums. Keeping this array
  // compact reduces shared-memory traffic and bank pressure.
  __shared__ float warp_sums[kTokensPerTile][kComputeWarps];
  __shared__ float alpha_shared;
  __shared__ float weights_shared[kTokensPerTile];

  const int pbs = static_cast<int>(page_block_size);

  static_assert(
      sizeof(TData) == 2,
      "CTA tile kernel assumes 16B chunks map to 8 elements for fp16/bf16.");
  constexpr int kChunkElems = 8;  // 8 * 2 bytes = 16 bytes.
  constexpr int CHUNKS = kHeadSize / kChunkElems;
  constexpr int LOADS_PER_TILE = CHUNKS * kTokensPerTile;

  // Multi-stage cp.async pipeline. Using >= 3 stages allows us to keep
  // multiple groups in-flight and overlap global->shared copies with compute.
  constexpr int kStages = 3;
  __shared__ __align__(16) TData sh_k[kStages][kTokensPerTile][kHeadSize];
  __shared__ __align__(16) TData sh_v[kStages][kTokensPerTile][kHeadSize];

  const float alibi_slope =
      (alibi_slopes == nullptr) ? 0.0f : alibi_slopes[head_idx];
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  int t_base = 0;
  for (int logical_block = 0; t_base < seqlen; ++logical_block, t_base += pbs) {
    const int physical_block = static_cast<int>(block_table[logical_block]);

    const TData* k_base =
        k_cache + physical_block * k_batch_stride + kv_head_idx * k_head_stride;
    const TData* v_base =
        v_cache + physical_block * v_batch_stride + kv_head_idx * v_head_stride;

    const int token_end = min(pbs, seqlen - t_base);
    const int num_tiles = (token_end + kTokensPerTile - 1) / kTokensPerTile;
    if (num_tiles <= 0) {
      continue;
    }

    int pending_groups = 0;
    const int preload = min(kStages, num_tiles);
    for (int ti = 0; ti < preload; ++ti) {
      const int token_in_block = ti * kTokensPerTile;
      const int tile_n = min(kTokensPerTile, token_end - token_in_block);
      for (int li = tid; li < LOADS_PER_TILE; li += kCtaThreads) {
        const int tok = li / CHUNKS;
        const int chunk = li - tok * CHUNKS;
        const int off = chunk * kChunkElems;
        if (tok < tile_n) {
          const TData* k_src =
              k_base + (token_in_block + tok) * k_row_stride + off;
          const TData* v_src =
              v_base + (token_in_block + tok) * v_row_stride + off;
          CpAsyncCaSharedGlobal16(&sh_k[ti][tok][off], k_src);
          CpAsyncCaSharedGlobal16(&sh_v[ti][tok][off], v_src);
        } else {
          reinterpret_cast<uint4*>(&sh_k[ti][tok][off])[0] =
              make_uint4(0, 0, 0, 0);
          reinterpret_cast<uint4*>(&sh_v[ti][tok][off])[0] =
              make_uint4(0, 0, 0, 0);
        }
      }
      CpAsyncCommit();
      ++pending_groups;
    }

    // Ensure tile 0 is ready. We want to keep up to (kStages - 1) groups
    // in flight for overlap, but still make forward progress in the tail
    // when we stop issuing new prefetch groups.
    int desired_pending = pending_groups - 1;
    if (desired_pending < 0) {
      desired_pending = 0;
    }
    if (desired_pending > (kStages - 1)) {
      desired_pending = (kStages - 1);
    }
    CpAsyncWaitGroupRt(desired_pending);
    pending_groups = desired_pending;
    if constexpr (NUM_WARPS == 1) {
      __syncwarp();
    } else {
      __syncthreads();
    }

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      const int buf = tile_idx % kStages;
      const int token_in_block = tile_idx * kTokensPerTile;
      const int tile_n = min(kTokensPerTile, token_end - token_in_block);

      float partial[kTokensPerTile];
#pragma unroll
      for (int j = 0; j < kTokensPerTile; ++j) {
        if (j < tile_n) {
          float k0 = 0.0f;
          float k1 = 0.0f;
          float k2 = 0.0f;
          float k3 = 0.0f;
#if defined(__CUDA_ARCH__)
          if constexpr (std::is_same_v<TData, half>) {
            if constexpr (kPack == 2) {
              const half2 kh2 =
                  *reinterpret_cast<const half2*>(&sh_k[buf][j][dim]);
              const float2 kf = __half22float2(kh2);
              k0 = kf.x;
              k1 = kf.y;
            } else {
              const half2 kh2_0 =
                  *reinterpret_cast<const half2*>(&sh_k[buf][j][dim + 0]);
              const half2 kh2_1 =
                  *reinterpret_cast<const half2*>(&sh_k[buf][j][dim + 2]);
              const float2 kf0 = __half22float2(kh2_0);
              const float2 kf1 = __half22float2(kh2_1);
              k0 = kf0.x;
              k1 = kf0.y;
              k2 = kf1.x;
              k3 = kf1.y;
            }
          } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
            if constexpr (kPack == 2) {
              const __nv_bfloat162 kb2 =
                  *reinterpret_cast<const __nv_bfloat162*>(&sh_k[buf][j][dim]);
              const float2 kf = __bfloat1622float2(kb2);
              k0 = kf.x;
              k1 = kf.y;
            } else {
              const __nv_bfloat162 kb2_0 =
                  *reinterpret_cast<const __nv_bfloat162*>(
                      &sh_k[buf][j][dim + 0]);
              const __nv_bfloat162 kb2_1 =
                  *reinterpret_cast<const __nv_bfloat162*>(
                      &sh_k[buf][j][dim + 2]);
              const float2 kf0 = __bfloat1622float2(kb2_0);
              const float2 kf1 = __bfloat1622float2(kb2_1);
              k0 = kf0.x;
              k1 = kf0.y;
              k2 = kf1.x;
              k3 = kf1.y;
            }
          } else
#endif
          {
            k0 = static_cast<float>(sh_k[buf][j][dim + 0]);
            k1 = static_cast<float>(sh_k[buf][j][dim + 1]);
            if constexpr (kPack == 4) {
              k2 = static_cast<float>(sh_k[buf][j][dim + 2]);
              k3 = static_cast<float>(sh_k[buf][j][dim + 3]);
            }
          }
          if constexpr (kPack == 2) {
            partial[j] = fmaf(q0, k0, q1 * k1);
          } else {
            partial[j] = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, q3 * k3)));
          }
        } else {
          partial[j] = 0.0f;
        }
      }

#pragma unroll
      for (int j = 0; j < kTokensPerTile; ++j) {
        float sum = WarpReduceSum(partial[j]);
        // Only compute warps contribute to qk; load-only warps would
        // otherwise write zeros and increase reduction overhead.
        if (lane == 0 && warp_id < kComputeWarps) {
          warp_sums[j][warp_id] = sum;
        }
      }

      if constexpr (NUM_WARPS == 1) {
        __syncwarp();
      } else {
        __syncthreads();
      }

      if (warp_id == 0) {
        // Distribute token-wise score computation across lanes to avoid
        // serial loops in lane0. kTokensPerTile <= 16 by construction.
        float score = -INFINITY;
        if (lane < kTokensPerTile && lane < tile_n) {
          float qk = 0.0f;
#pragma unroll
          for (int w = 0; w < kComputeWarps; ++w) {
            qk += warp_sums[lane][w];
          }
          const int t = t_base + token_in_block + lane;
          score = qk * scale_log2;
          if (alibi_slope != 0.0f) {
            score +=
                (alibi_slope * static_cast<float>(t - (seqlen - 1))) * kLog2e;
          }
        }

        float tile_max = WarpReduceMax(score);
        tile_max = __shfl_sync(0xffffffff, tile_max, 0);

        float m_new = 0.0f;
        if (lane == 0) {
          m_new = fmaxf(m, tile_max);
        }
        m_new = __shfl_sync(0xffffffff, m_new, 0);

        float w = 0.0f;
        if (lane < kTokensPerTile && lane < tile_n) {
          w = exp2f(score - m_new);
        }

        if (lane < kTokensPerTile) {
          weights_shared[lane] = (lane < tile_n) ? w : 0.0f;
        }

        float tile_sum = WarpReduceSum(w);
        if (lane == 0) {
          const float alpha = exp2f(m - m_new);
          alpha_shared = alpha;
          l = l * alpha + tile_sum;
          m = m_new;
        }
      }

      if constexpr (NUM_WARPS == 1) {
        __syncwarp();
      } else {
        __syncthreads();
      }

      const float alpha = alpha_shared;
      float sum_wv0 = 0.0f;
      float sum_wv1 = 0.0f;
      float sum_wv2 = 0.0f;
      float sum_wv3 = 0.0f;
#pragma unroll
      for (int j = 0; j < kTokensPerTile; ++j) {
        const float w = weights_shared[j];
        float v0 = 0.0f;
        float v1 = 0.0f;
        float v2 = 0.0f;
        float v3 = 0.0f;
#if defined(__CUDA_ARCH__)
        if constexpr (std::is_same_v<TData, half>) {
          if constexpr (kPack == 2) {
            const half2 vh2 =
                *reinterpret_cast<const half2*>(&sh_v[buf][j][dim]);
            const float2 vf = __half22float2(vh2);
            v0 = vf.x;
            v1 = vf.y;
          } else {
            const half2 vh2_0 =
                *reinterpret_cast<const half2*>(&sh_v[buf][j][dim + 0]);
            const half2 vh2_1 =
                *reinterpret_cast<const half2*>(&sh_v[buf][j][dim + 2]);
            const float2 vf0 = __half22float2(vh2_0);
            const float2 vf1 = __half22float2(vh2_1);
            v0 = vf0.x;
            v1 = vf0.y;
            v2 = vf1.x;
            v3 = vf1.y;
          }
        } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
          if constexpr (kPack == 2) {
            const __nv_bfloat162 vb2 =
                *reinterpret_cast<const __nv_bfloat162*>(&sh_v[buf][j][dim]);
            const float2 vf = __bfloat1622float2(vb2);
            v0 = vf.x;
            v1 = vf.y;
          } else {
            const __nv_bfloat162 vb2_0 =
                *reinterpret_cast<const __nv_bfloat162*>(
                    &sh_v[buf][j][dim + 0]);
            const __nv_bfloat162 vb2_1 =
                *reinterpret_cast<const __nv_bfloat162*>(
                    &sh_v[buf][j][dim + 2]);
            const float2 vf0 = __bfloat1622float2(vb2_0);
            const float2 vf1 = __bfloat1622float2(vb2_1);
            v0 = vf0.x;
            v1 = vf0.y;
            v2 = vf1.x;
            v3 = vf1.y;
          }
        } else
#endif
        {
          v0 = static_cast<float>(sh_v[buf][j][dim + 0]);
          v1 = static_cast<float>(sh_v[buf][j][dim + 1]);
          if constexpr (kPack == 4) {
            v2 = static_cast<float>(sh_v[buf][j][dim + 2]);
            v3 = static_cast<float>(sh_v[buf][j][dim + 3]);
          }
        }
        sum_wv0 = fmaf(w, v0, sum_wv0);
        sum_wv1 = fmaf(w, v1, sum_wv1);
        if constexpr (kPack == 4) {
          sum_wv2 = fmaf(w, v2, sum_wv2);
          sum_wv3 = fmaf(w, v3, sum_wv3);
        }
      }
      acc0 = acc0 * alpha + sum_wv0;
      acc1 = acc1 * alpha + sum_wv1;
      if constexpr (kPack == 4) {
        acc2 = acc2 * alpha + sum_wv2;
        acc3 = acc3 * alpha + sum_wv3;
      }

      // Prefetch the tile that will reuse this buffer (kStages steps ahead).
      const int prefetch_tile = tile_idx + kStages;
      if (prefetch_tile < num_tiles) {
        const int token_prefetch = prefetch_tile * kTokensPerTile;
        const int prefetch_n = min(kTokensPerTile, token_end - token_prefetch);
        for (int li = tid; li < LOADS_PER_TILE; li += kCtaThreads) {
          const int tok = li / CHUNKS;
          const int chunk = li - tok * CHUNKS;
          const int off = chunk * kChunkElems;
          if (tok < prefetch_n) {
            const TData* k_src =
                k_base + (token_prefetch + tok) * k_row_stride + off;
            const TData* v_src =
                v_base + (token_prefetch + tok) * v_row_stride + off;
            CpAsyncCaSharedGlobal16(&sh_k[buf][tok][off], k_src);
            CpAsyncCaSharedGlobal16(&sh_v[buf][tok][off], v_src);
          } else {
            reinterpret_cast<uint4*>(&sh_k[buf][tok][off])[0] =
                make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(&sh_v[buf][tok][off])[0] =
                make_uint4(0, 0, 0, 0);
          }
        }
        CpAsyncCommit();
        ++pending_groups;
      }

      if (tile_idx + 1 < num_tiles) {
        // Before consuming the next tile, ensure at least one group
        // completes. In steady state we keep (kStages - 1) in flight; in
        // the tail (no more prefetches) we gradually drain.
        int desired_pending = pending_groups - 1;
        if (desired_pending < 0) {
          desired_pending = 0;
        }
        if (desired_pending > (kStages - 1)) {
          desired_pending = (kStages - 1);
        }
        CpAsyncWaitGroupRt(desired_pending);
        pending_groups = desired_pending;
        if constexpr (NUM_WARPS == 1) {
          __syncwarp();
        } else {
          __syncthreads();
        }
      }
    }

    // Drain any in-flight async copies before moving to the next paged block.
    CpAsyncWaitAll();
    if constexpr (NUM_WARPS == 1) {
      __syncwarp();
    } else {
      __syncthreads();
    }
  }

  __shared__ float inv_l_shared;
  if (tid == 0) {
    inv_l_shared = 1.0f / (l + 1e-6f);
  }
  if constexpr (NUM_WARPS == 1) {
    __syncwarp();
  } else {
    __syncthreads();
  }

  {
    const float s = inv_l_shared;
    const float o0 = acc0 * s;
    const float o1 = acc1 * s;
    const float o2 = acc2 * s;
    const float o3 = acc3 * s;
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_same_v<TData, half>) {
      outptr[dim + 0] = __float2half_rn(o0);
      outptr[dim + 1] = __float2half_rn(o1);
      if constexpr (kPack == 4) {
        outptr[dim + 2] = __float2half_rn(o2);
        outptr[dim + 3] = __float2half_rn(o3);
      }
    } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
      outptr[dim + 0] = __float2bfloat16_rn(o0);
      outptr[dim + 1] = __float2bfloat16_rn(o1);
      if constexpr (kPack == 4) {
        outptr[dim + 2] = __float2bfloat16_rn(o2);
        outptr[dim + 3] = __float2bfloat16_rn(o3);
      }
    } else
#endif
    {
      outptr[dim + 0] = static_cast<TData>(o0);
      outptr[dim + 1] = static_cast<TData>(o1);
      if constexpr (kPack == 4) {
        outptr[dim + 2] = static_cast<TData>(o2);
        outptr[dim + 3] = static_cast<TData>(o3);
      }
    }
  }
}

// GQA/MQA fused decode kernel: one CTA computes outputs for kNGroups query
// heads that share the same KV head. This reduces redundant K/V reads when
// num_heads > num_kv_heads.
//
// v0.4: implemented for head_dim=128 and kNGroups=4 (common case: 32 Q heads /
// 8 KV heads).
template <typename TIndex, typename TData, int kHeadSize, int kCtaThreads,
          int kTokensPerTile, int kNGroups>
__device__ void FlashAttentionDecodeCtaGqaKernel(
    TData* out, const TData* q, const TData* k_cache, const TData* v_cache,
    const TIndex* block_tables, const TIndex* cache_lens,
    const float* alibi_slopes, size_t num_kv_heads, float scale,
    size_t max_num_blocks_per_seq, size_t page_block_size, ptrdiff_t qstride,
    ptrdiff_t k_batch_stride, ptrdiff_t k_row_stride, ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride, ptrdiff_t v_row_stride, ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {
  constexpr int kWarpSize = 32;
  static_assert(
      kHeadSize == 128,
      "v0.4 GQA fused CTA kernel is implemented for head_size=128 only.");
  static_assert(
      kNGroups == 4,
      "v0.4 GQA fused CTA kernel is implemented for kNGroups=4 only.");
  static_assert(kCtaThreads % kWarpSize == 0,
                "kCtaThreads must be a multiple of 32.");
  static_assert(kTokensPerTile > 0 && kTokensPerTile <= 16,
                "kTokensPerTile should stay small.");
  constexpr int NUM_WARPS = kCtaThreads / kWarpSize;

  // Pack dims per thread. For head_dim=128 and kCtaThreads=64, kPack=2.
  static_assert(kHeadSize % kCtaThreads == 0,
                "kHeadSize must be divisible by kCtaThreads.");
  constexpr int kPack = kHeadSize / kCtaThreads;
  static_assert(kPack == 2, "v0.4 GQA fused CTA kernel expects kPack=2.");
  constexpr int kPackedDims = kCtaThreads;
  constexpr int kComputeWarps = (kPackedDims + kWarpSize - 1) / kWarpSize;

  const int seqidx = blockIdx.y;
  const int kv_head_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int warp_id = tid / kWarpSize;
  const int dim = tid * kPack;

  const int seqlen = static_cast<int>(cache_lens[seqidx]);
  if (seqlen <= 0) {
    return;
  }

  // v0.4 limitation: alibi slopes are per query head; support can be added
  // later.
  if (alibi_slopes != nullptr) {
    return;
  }

  const TIndex* block_table =
      block_tables + seqidx * static_cast<int>(max_num_blocks_per_seq);

  // q/out are [num_seqs, num_heads, head_size]. For a KV head, we handle
  // kNGroups query heads: qhead = kv_head * kNGroups + g
  float q0[kNGroups];
  float q1[kNGroups];
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<TData, half>) {
#pragma unroll
    for (int g = 0; g < kNGroups; ++g) {
      const int qhead = kv_head_idx * kNGroups + g;
      const TData* qptr = q + seqidx * qstride + qhead * kHeadSize;
      const half2 qh2 = *reinterpret_cast<const half2*>(qptr + dim);
      const float2 qf = __half22float2(qh2);
      q0[g] = qf.x;
      q1[g] = qf.y;
    }
  } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
#pragma unroll
    for (int g = 0; g < kNGroups; ++g) {
      const int qhead = kv_head_idx * kNGroups + g;
      const TData* qptr = q + seqidx * qstride + qhead * kHeadSize;
      const __nv_bfloat162 qb2 =
          *reinterpret_cast<const __nv_bfloat162*>(qptr + dim);
      const float2 qf = __bfloat1622float2(qb2);
      q0[g] = qf.x;
      q1[g] = qf.y;
    }
  } else
#endif
  {
#pragma unroll
    for (int g = 0; g < kNGroups; ++g) {
      const int qhead = kv_head_idx * kNGroups + g;
      const TData* qptr = q + seqidx * qstride + qhead * kHeadSize;
      q0[g] = static_cast<float>(qptr[dim + 0]);
      q1[g] = static_cast<float>(qptr[dim + 1]);
    }
  }

  float acc0[kNGroups];
  float acc1[kNGroups];
  float m[kNGroups];
  float l[kNGroups];
#pragma unroll
  for (int g = 0; g < kNGroups; ++g) {
    acc0[g] = 0.0f;
    acc1[g] = 0.0f;
    m[g] = -INFINITY;
    l[g] = 0.0f;
  }

  __shared__ float warp_sums[kNGroups][kTokensPerTile][kComputeWarps];
  __shared__ float alpha_shared[kNGroups];
  __shared__ float weights_shared[kNGroups][kTokensPerTile];

  const int pbs = static_cast<int>(page_block_size);
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  static_assert(sizeof(TData) == 2, "CTA GQA kernel assumes fp16/bf16.");
  constexpr int kChunkElems = 8;  // 8 * 2 bytes = 16 bytes.
  constexpr int CHUNKS = kHeadSize / kChunkElems;
  constexpr int LOADS_PER_TILE = CHUNKS * kTokensPerTile;

  constexpr int kStages = 3;
  __shared__ __align__(16) TData sh_k[kStages][kTokensPerTile][kHeadSize];
  __shared__ __align__(16) TData sh_v[kStages][kTokensPerTile][kHeadSize];

  int t_base = 0;
  for (int logical_block = 0; t_base < seqlen; ++logical_block, t_base += pbs) {
    const int physical_block = static_cast<int>(block_table[logical_block]);

    const TData* k_base =
        k_cache + physical_block * k_batch_stride + kv_head_idx * k_head_stride;
    const TData* v_base =
        v_cache + physical_block * v_batch_stride + kv_head_idx * v_head_stride;

    const int token_end = min(pbs, seqlen - t_base);
    const int num_tiles = (token_end + kTokensPerTile - 1) / kTokensPerTile;
    if (num_tiles <= 0) {
      continue;
    }

    int pending_groups = 0;
    const int preload = min(kStages, num_tiles);
    for (int ti = 0; ti < preload; ++ti) {
      const int token_in_block = ti * kTokensPerTile;
      const int tile_n = min(kTokensPerTile, token_end - token_in_block);
      for (int li = tid; li < LOADS_PER_TILE; li += kCtaThreads) {
        const int tok = li / CHUNKS;
        const int chunk = li - tok * CHUNKS;
        const int off = chunk * kChunkElems;
        if (tok < tile_n) {
          const TData* k_src =
              k_base + (token_in_block + tok) * k_row_stride + off;
          const TData* v_src =
              v_base + (token_in_block + tok) * v_row_stride + off;
          CpAsyncCaSharedGlobal16(&sh_k[ti][tok][off], k_src);
          CpAsyncCaSharedGlobal16(&sh_v[ti][tok][off], v_src);
        } else {
          reinterpret_cast<uint4*>(&sh_k[ti][tok][off])[0] =
              make_uint4(0, 0, 0, 0);
          reinterpret_cast<uint4*>(&sh_v[ti][tok][off])[0] =
              make_uint4(0, 0, 0, 0);
        }
      }
      CpAsyncCommit();
      ++pending_groups;
    }

    int desired_pending = pending_groups - 1;
    if (desired_pending < 0) {
      desired_pending = 0;
    }
    if (desired_pending > (kStages - 1)) {
      desired_pending = (kStages - 1);
    }
    CpAsyncWaitGroupRt(desired_pending);
    pending_groups = desired_pending;
    if constexpr (NUM_WARPS == 1) {
      __syncwarp();
    } else {
      __syncthreads();
    }

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      const int buf = tile_idx % kStages;
      const int token_in_block = tile_idx * kTokensPerTile;
      const int tile_n = min(kTokensPerTile, token_end - token_in_block);

      // Compute QK partial sums for each group and each token in the tile.
      float partial_qk[kNGroups][kTokensPerTile];
#pragma unroll
      for (int g = 0; g < kNGroups; ++g) {
#pragma unroll
        for (int j = 0; j < kTokensPerTile; ++j) {
          if (j < tile_n) {
            float k0 = 0.0f;
            float k1 = 0.0f;
#if defined(__CUDA_ARCH__)
            if constexpr (std::is_same_v<TData, half>) {
              const half2 kh2 =
                  *reinterpret_cast<const half2*>(&sh_k[buf][j][dim]);
              const float2 kf = __half22float2(kh2);
              k0 = kf.x;
              k1 = kf.y;
            } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
              const __nv_bfloat162 kb2 =
                  *reinterpret_cast<const __nv_bfloat162*>(&sh_k[buf][j][dim]);
              const float2 kf = __bfloat1622float2(kb2);
              k0 = kf.x;
              k1 = kf.y;
            } else
#endif
            {
              k0 = static_cast<float>(sh_k[buf][j][dim + 0]);
              k1 = static_cast<float>(sh_k[buf][j][dim + 1]);
            }
            partial_qk[g][j] = fmaf(q0[g], k0, q1[g] * k1);
          } else {
            partial_qk[g][j] = 0.0f;
          }
        }
      }

#pragma unroll
      for (int g = 0; g < kNGroups; ++g) {
#pragma unroll
        for (int j = 0; j < kTokensPerTile; ++j) {
          const float sum = WarpReduceSum(partial_qk[g][j]);
          if (lane == 0 && warp_id < kComputeWarps) {
            warp_sums[g][j][warp_id] = sum;
          }
        }
      }

      if constexpr (NUM_WARPS == 1) {
        __syncwarp();
      } else {
        __syncthreads();
      }

      if (warp_id == 0) {
#pragma unroll
        for (int g = 0; g < kNGroups; ++g) {
          float score = -INFINITY;
          if (lane < kTokensPerTile && lane < tile_n) {
            float qk = 0.0f;
#pragma unroll
            for (int w = 0; w < kComputeWarps; ++w) {
              qk += warp_sums[g][lane][w];
            }
            score = qk * scale_log2;
          }

          float tile_max = WarpReduceMax(score);
          tile_max = __shfl_sync(0xffffffff, tile_max, 0);

          float m_new = 0.0f;
          if (lane == 0) {
            m_new = fmaxf(m[g], tile_max);
          }
          m_new = __shfl_sync(0xffffffff, m_new, 0);

          float w = 0.0f;
          if (lane < kTokensPerTile && lane < tile_n) {
            w = exp2f(score - m_new);
          }
          if (lane < kTokensPerTile) {
            weights_shared[g][lane] = (lane < tile_n) ? w : 0.0f;
          }

          const float tile_sum = WarpReduceSum(w);
          if (lane == 0) {
            const float alpha = exp2f(m[g] - m_new);
            alpha_shared[g] = alpha;
            l[g] = l[g] * alpha + tile_sum;
            m[g] = m_new;
          }
        }
      }

      if constexpr (NUM_WARPS == 1) {
        __syncwarp();
      } else {
        __syncthreads();
      }

      float alpha[kNGroups];
      float sum_wv0[kNGroups];
      float sum_wv1[kNGroups];
#pragma unroll
      for (int g = 0; g < kNGroups; ++g) {
        alpha[g] = alpha_shared[g];
        sum_wv0[g] = 0.0f;
        sum_wv1[g] = 0.0f;
      }

#pragma unroll
      for (int j = 0; j < kTokensPerTile; ++j) {
        float v0 = 0.0f;
        float v1 = 0.0f;
#if defined(__CUDA_ARCH__)
        if constexpr (std::is_same_v<TData, half>) {
          const half2 vh2 = *reinterpret_cast<const half2*>(&sh_v[buf][j][dim]);
          const float2 vf = __half22float2(vh2);
          v0 = vf.x;
          v1 = vf.y;
        } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
          const __nv_bfloat162 vb2 =
              *reinterpret_cast<const __nv_bfloat162*>(&sh_v[buf][j][dim]);
          const float2 vf = __bfloat1622float2(vb2);
          v0 = vf.x;
          v1 = vf.y;
        } else
#endif
        {
          v0 = static_cast<float>(sh_v[buf][j][dim + 0]);
          v1 = static_cast<float>(sh_v[buf][j][dim + 1]);
        }

#pragma unroll
        for (int g = 0; g < kNGroups; ++g) {
          const float w = weights_shared[g][j];
          sum_wv0[g] = fmaf(w, v0, sum_wv0[g]);
          sum_wv1[g] = fmaf(w, v1, sum_wv1[g]);
        }
      }

#pragma unroll
      for (int g = 0; g < kNGroups; ++g) {
        acc0[g] = acc0[g] * alpha[g] + sum_wv0[g];
        acc1[g] = acc1[g] * alpha[g] + sum_wv1[g];
      }

      const int prefetch_tile = tile_idx + kStages;
      if (prefetch_tile < num_tiles) {
        const int token_prefetch = prefetch_tile * kTokensPerTile;
        const int prefetch_n = min(kTokensPerTile, token_end - token_prefetch);
        for (int li = tid; li < LOADS_PER_TILE; li += kCtaThreads) {
          const int tok = li / CHUNKS;
          const int chunk = li - tok * CHUNKS;
          const int off = chunk * kChunkElems;
          if (tok < prefetch_n) {
            const TData* k_src =
                k_base + (token_prefetch + tok) * k_row_stride + off;
            const TData* v_src =
                v_base + (token_prefetch + tok) * v_row_stride + off;
            CpAsyncCaSharedGlobal16(&sh_k[buf][tok][off], k_src);
            CpAsyncCaSharedGlobal16(&sh_v[buf][tok][off], v_src);
          } else {
            reinterpret_cast<uint4*>(&sh_k[buf][tok][off])[0] =
                make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(&sh_v[buf][tok][off])[0] =
                make_uint4(0, 0, 0, 0);
          }
        }
        CpAsyncCommit();
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
        CpAsyncWaitGroupRt(desired_pending2);
        pending_groups = desired_pending2;
        if constexpr (NUM_WARPS == 1) {
          __syncwarp();
        } else {
          __syncthreads();
        }
      }
    }

    CpAsyncWaitAll();
    if constexpr (NUM_WARPS == 1) {
      __syncwarp();
    } else {
      __syncthreads();
    }
  }

  // Write outputs for each group.
  __shared__ float inv_l_shared[kNGroups];
  if (tid < kNGroups) {
    inv_l_shared[tid] = 1.0f / (l[tid] + 1e-6f);
  }
  if constexpr (NUM_WARPS == 1) {
    __syncwarp();
  } else {
    __syncthreads();
  }

#pragma unroll
  for (int g = 0; g < kNGroups; ++g) {
    const int qhead = kv_head_idx * kNGroups + g;
    TData* outptr = out + seqidx * o_stride + qhead * kHeadSize;
    const float s = inv_l_shared[g];
    const float o0 = acc0[g] * s;
    const float o1 = acc1[g] * s;
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_same_v<TData, half>) {
      outptr[dim + 0] = __float2half_rn(o0);
      outptr[dim + 1] = __float2half_rn(o1);
    } else if constexpr (std::is_same_v<TData, __nv_bfloat16>) {
      outptr[dim + 0] = __float2bfloat16_rn(o0);
      outptr[dim + 1] = __float2bfloat16_rn(o1);
    } else
#endif
    {
      outptr[dim + 0] = static_cast<TData>(o0);
      outptr[dim + 1] = static_cast<TData>(o1);
    }
  }
}

}  // namespace op::paged_attention::cuda

namespace infini::ops {

template <typename TIndex, typename TData, int kHeadSize>
__global__ void PagedAttentionInfinilmSplitKvCtaKernel(
    float* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l, const TData* __restrict__ q,
    const TData* __restrict__ k_cache, const TData* __restrict__ v_cache,
    const TIndex* __restrict__ block_tables, const TIndex* __restrict__ seqlens,
    const float* __restrict__ alibi_slopes, std::size_t num_kv_heads,
    float scale, std::size_t max_num_blocks_per_seq, std::size_t block_size,
    std::ptrdiff_t qstride, std::ptrdiff_t k_cacheblock_stride,
    std::ptrdiff_t k_cacheslot_stride, std::ptrdiff_t k_cachehead_stride,
    std::ptrdiff_t v_cacheblock_stride, std::ptrdiff_t v_cacheslot_stride,
    std::ptrdiff_t v_cachehead_stride, int num_splits) {
  op::paged_attention::cuda::FlashAttentionDecodeSplitKvCtaKernel<
      TIndex, TData, kHeadSize, 64, 8>(
      partial_acc, partial_m, partial_l, q, k_cache, v_cache, block_tables,
      seqlens, alibi_slopes, num_kv_heads, scale, max_num_blocks_per_seq,
      block_size, qstride, k_cacheblock_stride, k_cacheslot_stride,
      k_cachehead_stride, v_cacheblock_stride, v_cacheslot_stride,
      v_cachehead_stride, num_splits);
}

template <Device::Type kDev, typename TData, int kHeadSize>
__global__ void PagedAttentionInfinilmSplitKvCombineKernel(
    TData* __restrict__ out, const float* __restrict__ partial_acc,
    const float* __restrict__ partial_m, const float* __restrict__ partial_l,
    int num_splits, std::ptrdiff_t outstride) {
  op::paged_attention::cuda::FlashAttentionDecodeSplitKvCombineWarpKernel<
      kDev, TData, kHeadSize>(out, partial_acc, partial_m, partial_l,
                              num_splits, outstride);
}

template <typename TIndex, typename TData, int kHeadSize>
__global__ void PagedAttentionInfinilmDecodeWarpKernel(
    TData* __restrict__ out, const TData* __restrict__ q,
    const TData* __restrict__ k_cache, const TData* __restrict__ v_cache,
    const TIndex* __restrict__ block_tables, const TIndex* __restrict__ seqlens,
    const float* __restrict__ alibi_slopes, std::size_t num_heads,
    std::size_t num_kv_heads, float scale, std::size_t max_num_blocks_per_seq,
    std::size_t block_size, std::ptrdiff_t k_cacheblock_stride,
    std::ptrdiff_t k_cachehead_stride, std::ptrdiff_t k_cacheslot_stride,
    std::ptrdiff_t v_cacheblock_stride, std::ptrdiff_t v_cachehead_stride,
    std::ptrdiff_t v_cacheslot_stride, std::ptrdiff_t qstride,
    std::ptrdiff_t qhead_stride, std::ptrdiff_t outstride,
    std::ptrdiff_t outhead_stride, std::ptrdiff_t block_table_batch_stride,
    std::ptrdiff_t seqlens_stride) {
  constexpr int kWarpSize = 32;
  static_assert(kHeadSize == 64 || kHeadSize == 128,
                "PagedAttentionInfinilm decode supports head sizes 64 and 128");
  static_assert(kHeadSize % kWarpSize == 0,
                "head size must be divisible by 32");

  const int seqidx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int lane = threadIdx.x;
  constexpr int kDimsPerThread = kHeadSize / kWarpSize;
  constexpr float kLog2e = 1.4426950408889634f;

  __shared__ float reduce_buf[kWarpSize];
  __shared__ float state_buf[2];

  const int seqlen = static_cast<int>(seqlens[seqidx * seqlens_stride]);
  TData* outptr = out + seqidx * outstride + head_idx * outhead_stride;
  if (seqlen <= 0) {
#pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
      outptr[lane * kDimsPerThread + i] = static_cast<TData>(0.0f);
    }
    return;
  }

  const int queries_per_kv = static_cast<int>(num_heads / num_kv_heads);
  const int kv_head_idx = head_idx / queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.0f : alibi_slopes[head_idx];
  const float scale_log2 = scale * kLog2e;
  const TIndex* block_table = block_tables + seqidx * block_table_batch_stride;
  const TData* qptr = q + seqidx * qstride + head_idx * qhead_stride;

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
       t_base < seqlen &&
       logical_block < static_cast<int>(max_num_blocks_per_seq);
       ++logical_block, t_base += page_block_size) {
    const int physical_block = static_cast<int>(block_table[logical_block]);
    const TData* k_base = k_cache + physical_block * k_cacheblock_stride +
                          kv_head_idx * k_cachehead_stride;
    const TData* v_base = v_cache + physical_block * v_cacheblock_stride +
                          kv_head_idx * v_cachehead_stride;
    const int token_end = min(page_block_size, seqlen - t_base);

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
          score +=
              (alibi_slope * static_cast<float>(token_idx - (seqlen - 1))) *
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
