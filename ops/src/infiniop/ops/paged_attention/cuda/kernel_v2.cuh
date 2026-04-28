#ifndef __PAGED_ATTENTION_KERNEL_V2_CUH__
#define __PAGED_ATTENTION_KERNEL_V2_CUH__

namespace op::paged_attention::cuda {

struct OnlineSoftmaxState {
    float m = -INFINITY;
    float l = 0.0f;

    __device__ __forceinline__ void update(float x, float &alpha, float &beta) {
        const float m_new = fmaxf(m, x);
        alpha = expf(m - m_new);
        beta = expf(x - m_new);
        l = l * alpha + beta;
        m = m_new;
    }
};
__device__ __forceinline__ float warpReduceSum(float x) {
#if defined(ENABLE_ILUVATAR_API)
    // Iluvatar may use warp size 64; __shfl_sync(0xffffffff) only covers 32 threads.
    // Use shared-memory tree reduce for portability across warp sizes.
    constexpr int kMaxWarps = 16;
    __shared__ float _reduce_buf[kMaxWarps * 32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    _reduce_buf[threadIdx.x] = x;
    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (lane < offset) {
            _reduce_buf[warp_id * 32 + lane] += _reduce_buf[warp_id * 32 + lane + offset];
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

__device__ __forceinline__ float warpBroadcast(float x, int src_lane) {
#if defined(ENABLE_ILUVATAR_API)
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

__device__ __forceinline__ float warpReduceMax(float x) {
#if defined(ENABLE_ILUVATAR_API)
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

__device__ __forceinline__ unsigned int cvtaToShared(const void *ptr) {
#if defined(ENABLE_ILUVATAR_API)
    return static_cast<unsigned int>(reinterpret_cast<uintptr_t>(ptr));
#else
    return static_cast<unsigned int>(__cvta_generic_to_shared(ptr));
#endif
}

__device__ __forceinline__ void cpAsyncCaSharedGlobal16(void *dst_shared, const void *src_global) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const unsigned int dst = cvtaToShared(dst_shared);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src_global));
#else
    auto *dst = reinterpret_cast<uint4 *>(dst_shared);
    const auto *src = reinterpret_cast<const uint4 *>(src_global);
    *dst = *src;
#endif
}

__device__ __forceinline__ void cpAsyncCommit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cpAsyncWaitGroup() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// cp.async.wait_group requires a compile-time immediate, so for small fixed
// stage counts we provide a tiny runtime switch.
__device__ __forceinline__ void cpAsyncWaitGroupRt(int n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    if (n <= 0) {
        cpAsyncWaitGroup<0>();
    } else if (n == 1) {
        cpAsyncWaitGroup<1>();
    } else {
        // Clamp to 2 because v0.4 CTA kernel uses STAGES=3.
        cpAsyncWaitGroup<2>();
    }
#else
    (void)n;
#endif
}

__device__ __forceinline__ void cpAsyncWaitAll() {
    cpAsyncWaitGroup<0>();
}

template <typename Tindex, typename Tdata, int HEAD_SIZE>
__device__ void flashAttentionDecodeWarpKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *cache_lens_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {

    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int lane = threadIdx.x;
    constexpr int kWarpSize = 32;
    static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128, "Only head_size 64/128 supported in v0.4.");
    static_assert(HEAD_SIZE % kWarpSize == 0, "HEAD_SIZE must be divisible by 32.");
    constexpr int DIMS_PER_THREAD = HEAD_SIZE / kWarpSize;

    const int seq_len = static_cast<int>(cache_lens_[seq_idx]);
    if (seq_len <= 0) {
        return;
    }

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    const Tindex *block_table = block_tables_ + seq_idx * static_cast<int>(max_num_blocks_per_seq);

    // q/out are [num_seqs, num_heads, head_size]
    const Tdata *q_ptr = q_ + seq_idx * q_stride + head_idx * HEAD_SIZE;
    Tdata *out_ptr = out_ + seq_idx * o_stride + head_idx * HEAD_SIZE;

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

    // Iterate by blocks to avoid per-token division/mod and redundant block_table loads.
    // Note: Per-token cp.async prefetching is generally too fine-grained for decode and can regress.
    // We keep the warp kernel simple and reserve cp.async pipelining for CTA tile kernels.
    int t_base = 0;
    for (int logical_block = 0; t_base < seq_len; ++logical_block, t_base += pbs) {
        int physical_block = 0;
        if (lane == 0) {
            physical_block = static_cast<int>(block_table[logical_block]);
        }
        physical_block = __shfl_sync(0xffffffff, physical_block, 0);

        const Tdata *k_base = k_cache_ + physical_block * k_batch_stride + kv_head_idx * k_head_stride;
        const Tdata *v_base = v_cache_ + physical_block * v_batch_stride + kv_head_idx * v_head_stride;

        const int token_end = min(pbs, seq_len - t_base);
        for (int token_in_block = 0; token_in_block < token_end; ++token_in_block) {
            const int t = t_base + token_in_block;
            const Tdata *k_ptr = k_base + token_in_block * k_row_stride;
            const Tdata *v_ptr = v_base + token_in_block * v_row_stride;

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

            qk = warpReduceSum(qk);

            float alpha = 1.0f;
            float beta = 0.0f;
            if (lane == 0) {
                float score = qk * scale_log2;
                if (alibi_slope != 0.0f) {
                    score += (alibi_slope * static_cast<float>(t - (seq_len - 1))) * kLog2e;
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
    inv_l = __shfl_sync(0xffffffff, inv_l, 0);

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

// Split-KV decode (FA2-style): each split scans a shard of KV and writes partial (m, l, acc)
// to workspace, then a combine kernel merges splits into final out.
template <typename Tindex, typename Tdata, int HEAD_SIZE>
__device__ void flashAttentionDecodeSplitKvWarpKernel(
    float *partial_acc, // [num_splits, num_seqs, num_heads, head_size]
    float *partial_m,   // [num_splits, num_seqs, num_heads]
    float *partial_l,   // [num_splits, num_seqs, num_heads]
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *cache_lens_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    int num_splits) {

    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int split_idx = static_cast<int>(blockIdx.z);
    const int lane = threadIdx.x;
    constexpr int kWarpSize = 32;
    static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128, "Only head_size 64/128 supported in v0.4.");
    static_assert(HEAD_SIZE % kWarpSize == 0, "HEAD_SIZE must be divisible by 32.");
    constexpr int DIMS_PER_THREAD = HEAD_SIZE / kWarpSize;

    const int seq_len = static_cast<int>(cache_lens_[seq_idx]);
    if (seq_len <= 0 || num_splits <= 0) {
        return;
    }

    // Split the [0, seq_len) range into num_splits contiguous shards.
    const int shard = (seq_len + num_splits - 1) / num_splits;
    const int start = split_idx * shard;
    const int end = min(seq_len, start + shard);
    if (start >= end) {
        // Empty shard => write neutral element.
        const int n = gridDim.y * gridDim.x;
        const int idx = (split_idx * n + seq_idx * gridDim.x + head_idx);
        if (lane == 0) {
            partial_m[idx] = -INFINITY;
            partial_l[idx] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; ++i) {
            const int dim = lane * DIMS_PER_THREAD + i;
            partial_acc[idx * HEAD_SIZE + dim] = 0.0f;
        }
        return;
    }

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    const Tindex *block_table = block_tables_ + seq_idx * static_cast<int>(max_num_blocks_per_seq);
    const Tdata *q_ptr = q_ + seq_idx * q_stride + head_idx * HEAD_SIZE;

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

        const Tdata *k_base = k_cache_ + physical_block * k_batch_stride + kv_head_idx * k_head_stride;
        const Tdata *v_base = v_cache_ + physical_block * v_batch_stride + kv_head_idx * v_head_stride;

        const int token_end = min(pbs, end - logical_block * pbs);
        for (; token_in_block < token_end && t < end; ++token_in_block, ++t) {
            const Tdata *k_ptr = k_base + token_in_block * k_row_stride;
            const Tdata *v_ptr = v_base + token_in_block * v_row_stride;

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

            qk = warpReduceSum(qk);

            float alpha = 1.0f;
            float beta = 0.0f;
            if (lane == 0) {
                float score = qk * scale_log2;
                if (alibi_slope != 0.0f) {
                    score += (alibi_slope * static_cast<float>(t - (seq_len - 1))) * kLog2e;
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
        token_in_block = 0;
    }

    const int n = gridDim.y * gridDim.x;
    const int idx = (split_idx * n + seq_idx * gridDim.x + head_idx);
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

template <typename Tdata, int HEAD_SIZE>
__device__ void flashAttentionDecodeSplitKvCombineWarpKernel(
    Tdata *out_,
    const float *partial_acc, // [num_splits, num_seqs, num_heads, head_size]
    const float *partial_m,   // [num_splits, num_seqs, num_heads]
    const float *partial_l,   // [num_splits, num_seqs, num_heads]
    int num_splits,
    ptrdiff_t o_stride) {

    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int lane = threadIdx.x;
    constexpr int kWarpSize = 32;
    static_assert(HEAD_SIZE % kWarpSize == 0, "HEAD_SIZE must be divisible by 32.");
    constexpr int DIMS_PER_THREAD = HEAD_SIZE / kWarpSize;

    const int n = gridDim.y * gridDim.x;
    const int base = (seq_idx * gridDim.x + head_idx);

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
    Tdata *out_ptr = out_ + seq_idx * o_stride + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; ++i) {
        const int dim = lane * DIMS_PER_THREAD + i;
        float acc = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            const float ms = partial_m[s * n + base];
            const float w = exp2f(ms - m);
            acc += partial_acc[(s * n + base) * HEAD_SIZE + dim] * w;
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

// Split-KV decode with a CTA tile kernel (FA2-style): each CTA scans a shard of KV,
// writes partial (m, l, acc) to workspace, then a combine kernel merges splits.
template <typename Tindex, typename Tdata, int HEAD_SIZE, int CTA_THREADS, int TOKENS_PER_TILE>
__device__ void flashAttentionDecodeSplitKvCtaKernel(
    float *partial_acc, // [num_splits, num_seqs, num_heads, head_size]
    float *partial_m,   // [num_splits, num_seqs, num_heads]
    float *partial_l,   // [num_splits, num_seqs, num_heads]
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *cache_lens_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    int num_splits) {

    constexpr int kWarpSize = 32;
    static_assert(CTA_THREADS % kWarpSize == 0, "CTA_THREADS must be a multiple of 32.");
    static_assert(TOKENS_PER_TILE > 0 && TOKENS_PER_TILE <= 16, "TOKENS_PER_TILE should stay small.");
    constexpr int NUM_WARPS = CTA_THREADS / kWarpSize;

    static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128, "Only head_size 64/128 supported in v0.4.");
    static_assert(HEAD_SIZE % CTA_THREADS == 0, "HEAD_SIZE must be divisible by CTA_THREADS.");
    constexpr int kPack = HEAD_SIZE / CTA_THREADS; // 2 (64@32t, 128@64t) or 4 (128@32t)
    static_assert(kPack == 2 || kPack == 4, "v0.4 split-kv CTA kernel supports kPack=2/4 only.");
    constexpr int kPackedDims = CTA_THREADS;
    constexpr int kComputeWarps = (kPackedDims + kWarpSize - 1) / kWarpSize;

    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int split_idx = static_cast<int>(blockIdx.z);
    const int tid = threadIdx.x;
    const int lane = tid % kWarpSize;
    const int warp_id = tid / kWarpSize;

    const int seq_len = static_cast<int>(cache_lens_[seq_idx]);
    if (seq_len <= 0 || num_splits <= 0) {
        return;
    }

    // Split the [0, seq_len) range into num_splits contiguous shards.
    const int shard = (seq_len + num_splits - 1) / num_splits;
    const int start = split_idx * shard;
    const int end = min(seq_len, start + shard);

    const int n = gridDim.y * gridDim.x;
    const int idx = (split_idx * n + seq_idx * gridDim.x + head_idx);

    if (start >= end) {
        // Empty shard => write neutral element.
        if (tid == 0) {
            partial_m[idx] = -INFINITY;
            partial_l[idx] = 0.0f;
        }
        const int dim = tid * kPack;
        if constexpr (kPack == 2) {
            partial_acc[idx * HEAD_SIZE + dim + 0] = 0.0f;
            partial_acc[idx * HEAD_SIZE + dim + 1] = 0.0f;
        } else {
            partial_acc[idx * HEAD_SIZE + dim + 0] = 0.0f;
            partial_acc[idx * HEAD_SIZE + dim + 1] = 0.0f;
            partial_acc[idx * HEAD_SIZE + dim + 2] = 0.0f;
            partial_acc[idx * HEAD_SIZE + dim + 3] = 0.0f;
        }
        return;
    }

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const Tindex *block_table = block_tables_ + seq_idx * static_cast<int>(max_num_blocks_per_seq);
    const Tdata *q_ptr = q_ + seq_idx * q_stride + head_idx * HEAD_SIZE;

    const int dim = tid * kPack;
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_same_v<Tdata, half>) {
        if constexpr (kPack == 2) {
            const half2 qh2 = *reinterpret_cast<const half2 *>(q_ptr + dim);
            const float2 qf = __half22float2(qh2);
            q0 = qf.x;
            q1 = qf.y;
        } else {
            const half2 qh2_0 = *reinterpret_cast<const half2 *>(q_ptr + dim + 0);
            const half2 qh2_1 = *reinterpret_cast<const half2 *>(q_ptr + dim + 2);
            const float2 qf0 = __half22float2(qh2_0);
            const float2 qf1 = __half22float2(qh2_1);
            q0 = qf0.x;
            q1 = qf0.y;
            q2 = qf1.x;
            q3 = qf1.y;
        }
    } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
        if constexpr (kPack == 2) {
            const __nv_bfloat162 qb2 = *reinterpret_cast<const __nv_bfloat162 *>(q_ptr + dim);
            const float2 qf = __bfloat1622float2(qb2);
            q0 = qf.x;
            q1 = qf.y;
        } else {
            const __nv_bfloat162 qb2_0 = *reinterpret_cast<const __nv_bfloat162 *>(q_ptr + dim + 0);
            const __nv_bfloat162 qb2_1 = *reinterpret_cast<const __nv_bfloat162 *>(q_ptr + dim + 2);
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
        q0 = static_cast<float>(q_ptr[dim + 0]);
        q1 = static_cast<float>(q_ptr[dim + 1]);
        if constexpr (kPack == 4) {
            q2 = static_cast<float>(q_ptr[dim + 2]);
            q3 = static_cast<float>(q_ptr[dim + 3]);
        }
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    float m = -INFINITY;
    float l = 0.0f;

    __shared__ float warp_sums[TOKENS_PER_TILE][kComputeWarps];
    __shared__ float alpha_shared;
    __shared__ float weights_shared[TOKENS_PER_TILE];

    const int pbs = static_cast<int>(page_block_size);
    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    static_assert(sizeof(Tdata) == 2, "CTA split-kv kernel assumes fp16/bf16.");
    constexpr int CHUNK_ELEMS = 8; // 8 * 2 bytes = 16 bytes.
    constexpr int CHUNKS = HEAD_SIZE / CHUNK_ELEMS;
    constexpr int LOADS_PER_TILE = CHUNKS * TOKENS_PER_TILE;

    constexpr int STAGES = 3;
    __shared__ __align__(16) Tdata sh_k[STAGES][TOKENS_PER_TILE][HEAD_SIZE];
    __shared__ __align__(16) Tdata sh_v[STAGES][TOKENS_PER_TILE][HEAD_SIZE];

    const int first_block = start / pbs;
    const int last_block = (end - 1) / pbs;

    for (int logical_block = first_block; logical_block <= last_block; ++logical_block) {
        const int physical_block = static_cast<int>(block_table[logical_block]);
        const Tdata *k_base = k_cache_ + physical_block * k_batch_stride + kv_head_idx * k_head_stride;
        const Tdata *v_base = v_cache_ + physical_block * v_batch_stride + kv_head_idx * v_head_stride;

        const int t_base = logical_block * pbs;
        const int token_begin = (logical_block == first_block) ? (start - t_base) : 0;
        const int token_end = (logical_block == last_block) ? (end - t_base) : pbs;
        const int token_count = token_end - token_begin;
        if (token_count <= 0) {
            continue;
        }

        const int num_tiles = (token_count + TOKENS_PER_TILE - 1) / TOKENS_PER_TILE;
        int pending_groups = 0;
        const int preload = min(STAGES, num_tiles);
        for (int ti = 0; ti < preload; ++ti) {
            const int token_in_block = token_begin + ti * TOKENS_PER_TILE;
            const int tile_n = min(TOKENS_PER_TILE, token_end - token_in_block);
            for (int li = tid; li < LOADS_PER_TILE; li += CTA_THREADS) {
                const int tok = li / CHUNKS;
                const int chunk = li - tok * CHUNKS;
                const int off = chunk * CHUNK_ELEMS;
                if (tok < tile_n) {
                    const Tdata *k_src = k_base + (token_in_block + tok) * k_row_stride + off;
                    const Tdata *v_src = v_base + (token_in_block + tok) * v_row_stride + off;
                    cpAsyncCaSharedGlobal16(&sh_k[ti][tok][off], k_src);
                    cpAsyncCaSharedGlobal16(&sh_v[ti][tok][off], v_src);
                } else {
                    reinterpret_cast<uint4 *>(&sh_k[ti][tok][off])[0] = make_uint4(0, 0, 0, 0);
                    reinterpret_cast<uint4 *>(&sh_v[ti][tok][off])[0] = make_uint4(0, 0, 0, 0);
                }
            }
            cpAsyncCommit();
            ++pending_groups;
        }

        int desired_pending = pending_groups - 1;
        if (desired_pending < 0) {
            desired_pending = 0;
        }
        if (desired_pending > (STAGES - 1)) {
            desired_pending = (STAGES - 1);
        }
        cpAsyncWaitGroupRt(desired_pending);
        pending_groups = desired_pending;
        if constexpr (NUM_WARPS == 1) {
            __syncwarp();
        } else {
            __syncthreads();
        }

        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            const int buf = tile_idx % STAGES;
            const int token_in_block = token_begin + tile_idx * TOKENS_PER_TILE;
            const int tile_n = min(TOKENS_PER_TILE, token_end - token_in_block);

            float partial[TOKENS_PER_TILE];
#pragma unroll
            for (int j = 0; j < TOKENS_PER_TILE; ++j) {
                if (j < tile_n) {
                    float k0 = 0.0f, k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;
#if defined(__CUDA_ARCH__)
                    if constexpr (std::is_same_v<Tdata, half>) {
                        if constexpr (kPack == 2) {
                            const half2 kh2 = *reinterpret_cast<const half2 *>(&sh_k[buf][j][dim]);
                            const float2 kf = __half22float2(kh2);
                            k0 = kf.x;
                            k1 = kf.y;
                        } else {
                            const half2 kh2_0 = *reinterpret_cast<const half2 *>(&sh_k[buf][j][dim + 0]);
                            const half2 kh2_1 = *reinterpret_cast<const half2 *>(&sh_k[buf][j][dim + 2]);
                            const float2 kf0 = __half22float2(kh2_0);
                            const float2 kf1 = __half22float2(kh2_1);
                            k0 = kf0.x;
                            k1 = kf0.y;
                            k2 = kf1.x;
                            k3 = kf1.y;
                        }
                    } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                        if constexpr (kPack == 2) {
                            const __nv_bfloat162 kb2 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_k[buf][j][dim]);
                            const float2 kf = __bfloat1622float2(kb2);
                            k0 = kf.x;
                            k1 = kf.y;
                        } else {
                            const __nv_bfloat162 kb2_0 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_k[buf][j][dim + 0]);
                            const __nv_bfloat162 kb2_1 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_k[buf][j][dim + 2]);
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
            for (int j = 0; j < TOKENS_PER_TILE; ++j) {
                const float sum = warpReduceSum(partial[j]);
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
                if (lane < TOKENS_PER_TILE && lane < tile_n) {
                    float qk = 0.0f;
#pragma unroll
                    for (int w = 0; w < kComputeWarps; ++w) {
                        qk += warp_sums[lane][w];
                    }
                    const int t = t_base + token_in_block + lane;
                    score = qk * scale_log2;
                    if (alibi_slope != 0.0f) {
                        score += (alibi_slope * static_cast<float>(t - (seq_len - 1))) * kLog2e;
                    }
                }

                float tile_max = warpReduceMax(score);
                tile_max = __shfl_sync(0xffffffff, tile_max, 0);

                float m_new = 0.0f;
                if (lane == 0) {
                    m_new = fmaxf(m, tile_max);
                }
                m_new = __shfl_sync(0xffffffff, m_new, 0);

                float w = 0.0f;
                if (lane < TOKENS_PER_TILE && lane < tile_n) {
                    w = exp2f(score - m_new);
                }
                if (lane < TOKENS_PER_TILE) {
                    weights_shared[lane] = (lane < tile_n) ? w : 0.0f;
                }

                const float tile_sum = warpReduceSum(w);
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
            for (int j = 0; j < TOKENS_PER_TILE; ++j) {
                const float w = weights_shared[j];
                float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
#if defined(__CUDA_ARCH__)
                if constexpr (std::is_same_v<Tdata, half>) {
                    if constexpr (kPack == 2) {
                        const half2 vh2 = *reinterpret_cast<const half2 *>(&sh_v[buf][j][dim]);
                        const float2 vf = __half22float2(vh2);
                        v0 = vf.x;
                        v1 = vf.y;
                    } else {
                        const half2 vh2_0 = *reinterpret_cast<const half2 *>(&sh_v[buf][j][dim + 0]);
                        const half2 vh2_1 = *reinterpret_cast<const half2 *>(&sh_v[buf][j][dim + 2]);
                        const float2 vf0 = __half22float2(vh2_0);
                        const float2 vf1 = __half22float2(vh2_1);
                        v0 = vf0.x;
                        v1 = vf0.y;
                        v2 = vf1.x;
                        v3 = vf1.y;
                    }
                } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                    if constexpr (kPack == 2) {
                        const __nv_bfloat162 vb2 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_v[buf][j][dim]);
                        const float2 vf = __bfloat1622float2(vb2);
                        v0 = vf.x;
                        v1 = vf.y;
                    } else {
                        const __nv_bfloat162 vb2_0 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_v[buf][j][dim + 0]);
                        const __nv_bfloat162 vb2_1 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_v[buf][j][dim + 2]);
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

            const int prefetch_tile = tile_idx + STAGES;
            if (prefetch_tile < num_tiles) {
                const int token_prefetch = token_begin + prefetch_tile * TOKENS_PER_TILE;
                const int prefetch_n = min(TOKENS_PER_TILE, token_end - token_prefetch);
                for (int li = tid; li < LOADS_PER_TILE; li += CTA_THREADS) {
                    const int tok = li / CHUNKS;
                    const int chunk = li - tok * CHUNKS;
                    const int off = chunk * CHUNK_ELEMS;
                    if (tok < prefetch_n) {
                        const Tdata *k_src = k_base + (token_prefetch + tok) * k_row_stride + off;
                        const Tdata *v_src = v_base + (token_prefetch + tok) * v_row_stride + off;
                        cpAsyncCaSharedGlobal16(&sh_k[buf][tok][off], k_src);
                        cpAsyncCaSharedGlobal16(&sh_v[buf][tok][off], v_src);
                    } else {
                        reinterpret_cast<uint4 *>(&sh_k[buf][tok][off])[0] = make_uint4(0, 0, 0, 0);
                        reinterpret_cast<uint4 *>(&sh_v[buf][tok][off])[0] = make_uint4(0, 0, 0, 0);
                    }
                }
                cpAsyncCommit();
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
                cpAsyncWaitGroupRt(desired_pending2);
                pending_groups = desired_pending2;
                if constexpr (NUM_WARPS == 1) {
                    __syncwarp();
                } else {
                    __syncthreads();
                }
            }
        }

        cpAsyncWaitAll();
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
        partial_acc[idx * HEAD_SIZE + dim + 0] = acc0;
        partial_acc[idx * HEAD_SIZE + dim + 1] = acc1;
    } else {
        partial_acc[idx * HEAD_SIZE + dim + 0] = acc0;
        partial_acc[idx * HEAD_SIZE + dim + 1] = acc1;
        partial_acc[idx * HEAD_SIZE + dim + 2] = acc2;
        partial_acc[idx * HEAD_SIZE + dim + 3] = acc3;
    }
}

template <typename Tindex, typename Tdata, int HEAD_SIZE>
__device__ void flashAttentionDecodeCtaPipelinedKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *cache_lens_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {

    constexpr int kWarpSize = 32;
    static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128, "Only head_size 64/128 supported in v0.4.");
    static_assert(HEAD_SIZE % kWarpSize == 0, "HEAD_SIZE must be divisible by 32.");
    constexpr int NUM_WARPS = HEAD_SIZE / kWarpSize;

    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % kWarpSize;
    const int warp_id = tid / kWarpSize;

    const int seq_len = static_cast<int>(cache_lens_[seq_idx]);
    if (seq_len <= 0) {
        return;
    }

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    const Tindex *block_table = block_tables_ + seq_idx * static_cast<int>(max_num_blocks_per_seq);

    const Tdata *q_ptr = q_ + seq_idx * q_stride + head_idx * HEAD_SIZE;
    Tdata *out_ptr = out_ + seq_idx * o_stride + head_idx * HEAD_SIZE;

    const float q_val = static_cast<float>(q_ptr[tid]);
    float acc = 0.0f;

    float m = -INFINITY;
    float l = 0.0f;

    __shared__ Tdata sh_k[2][HEAD_SIZE];
    __shared__ Tdata sh_v[2][HEAD_SIZE];
    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float alpha_s;
    __shared__ float beta_s;
    __shared__ int physical_block_s;
    constexpr int CHUNK_ELEMS = 8; // 8 * 2 bytes = 16 bytes.
    constexpr int CHUNKS = HEAD_SIZE / CHUNK_ELEMS;

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
        const Tdata *k_base = k_cache_ + physical_block_s * k_batch_stride + kv_head_idx * k_head_stride;
        const Tdata *v_base = v_cache_ + physical_block_s * v_batch_stride + kv_head_idx * v_head_stride;
        if (tid < CHUNKS) {
            const int off = tid * CHUNK_ELEMS;
            cpAsyncCaSharedGlobal16(&sh_k[buf][off], (k_base + 0 * k_row_stride) + off);
            cpAsyncCaSharedGlobal16(&sh_v[buf][off], (v_base + 0 * v_row_stride) + off);
        }
        cpAsyncCommit();
        cpAsyncWaitAll();
        __syncthreads();
    }

    for (int t = 0; t < seq_len; ++t) {
        // Compute current token location within paged KV.
        const int next_t = t + 1;
        const bool has_next = next_t < seq_len;

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

            const Tdata *k_base = k_cache_ + physical_block_s * k_batch_stride + kv_head_idx * k_head_stride;
            const Tdata *v_base = v_cache_ + physical_block_s * v_batch_stride + kv_head_idx * v_head_stride;
            const Tdata *k_src = k_base + next_in_block * k_row_stride;
            const Tdata *v_src = v_base + next_in_block * v_row_stride;
            if (tid < CHUNKS) {
                const int off = tid * CHUNK_ELEMS;
                cpAsyncCaSharedGlobal16(&sh_k[buf ^ 1][off], k_src + off);
                cpAsyncCaSharedGlobal16(&sh_v[buf ^ 1][off], v_src + off);
            }
            cpAsyncCommit();
        }

        // Dot: each thread handles one dim, reduce across head dim.
        const float k_val = static_cast<float>(sh_k[buf][tid]);
        float partial = q_val * k_val;
        float warp_sum = warpReduceSum(partial);
        if (lane == 0) {
            warp_sums[warp_id] = warp_sum;
        }
        __syncthreads();

        float qk = 0.0f;
        if (warp_id == 0) {
            float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
            v = warpReduceSum(v);
            if (lane == 0) {
                qk = v;
                float score = qk * scale_log2;
                if (alibi_slope != 0.0f) {
                    score += (alibi_slope * static_cast<float>(t - (seq_len - 1))) * kLog2e;
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
            cpAsyncWaitAll();
            __syncthreads();
            buf ^= 1;
        }
    }

    __shared__ float inv_l_s;
    if (tid == 0) {
        inv_l_s = 1.0f / (l + 1e-6f);
    }
    __syncthreads();
    out_ptr[tid] = static_cast<Tdata>(acc * inv_l_s);
}

template <typename Tindex, typename Tdata, int HEAD_SIZE, int CTA_THREADS, int TOKENS_PER_TILE>
__device__ void flashAttentionDecodeCtaKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *cache_lens_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {

    constexpr int kWarpSize = 32;
    static_assert(CTA_THREADS % kWarpSize == 0, "CTA_THREADS must be a multiple of 32.");
    static_assert(TOKENS_PER_TILE > 0 && TOKENS_PER_TILE <= 16, "TOKENS_PER_TILE should stay small.");
    constexpr int NUM_WARPS = CTA_THREADS / kWarpSize;

    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % kWarpSize;
    const int warp_id = tid / kWarpSize;

    // Each thread owns a small packed vector of head dims. This lets us shrink the
    // CTA to 1-2 warps and reduce block-wide synchronization overhead.
    static_assert(HEAD_SIZE % CTA_THREADS == 0, "HEAD_SIZE must be divisible by CTA_THREADS.");
    constexpr int kPack = HEAD_SIZE / CTA_THREADS; // 2 (64@32t, 128@64t) or 4 (128@32t)
    static_assert(kPack == 2 || kPack == 4, "v0.4 CTA tile kernel supports kPack=2/4 only.");
    constexpr int kPackedDims = CTA_THREADS;
    constexpr int kComputeWarps = (kPackedDims + kWarpSize - 1) / kWarpSize;
    const int dim = tid * kPack;

    const int seq_len = static_cast<int>(cache_lens_[seq_idx]);
    if (seq_len <= 0) {
        return;
    }

    const int num_heads = gridDim.x;
    const int num_queries_per_kv = num_heads / static_cast<int>(num_kv_heads);
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const Tindex *block_table = block_tables_ + seq_idx * static_cast<int>(max_num_blocks_per_seq);

    // q/out are [num_seqs, num_heads, head_size]
    const Tdata *q_ptr = q_ + seq_idx * q_stride + head_idx * HEAD_SIZE;
    Tdata *out_ptr = out_ + seq_idx * o_stride + head_idx * HEAD_SIZE;

    float q0 = 0.0f;
    float q1 = 0.0f;
    float q2 = 0.0f;
    float q3 = 0.0f;
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_same_v<Tdata, half>) {
        if constexpr (kPack == 2) {
            const half2 qh2 = *reinterpret_cast<const half2 *>(q_ptr + dim);
            const float2 qf = __half22float2(qh2);
            q0 = qf.x;
            q1 = qf.y;
        } else {
            const half2 qh2_0 = *reinterpret_cast<const half2 *>(q_ptr + dim + 0);
            const half2 qh2_1 = *reinterpret_cast<const half2 *>(q_ptr + dim + 2);
            const float2 qf0 = __half22float2(qh2_0);
            const float2 qf1 = __half22float2(qh2_1);
            q0 = qf0.x;
            q1 = qf0.y;
            q2 = qf1.x;
            q3 = qf1.y;
        }
    } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
        if constexpr (kPack == 2) {
            const __nv_bfloat162 qb2 = *reinterpret_cast<const __nv_bfloat162 *>(q_ptr + dim);
            const float2 qf = __bfloat1622float2(qb2);
            q0 = qf.x;
            q1 = qf.y;
        } else {
            const __nv_bfloat162 qb2_0 = *reinterpret_cast<const __nv_bfloat162 *>(q_ptr + dim + 0);
            const __nv_bfloat162 qb2_1 = *reinterpret_cast<const __nv_bfloat162 *>(q_ptr + dim + 2);
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
        q0 = static_cast<float>(q_ptr[dim + 0]);
        q1 = static_cast<float>(q_ptr[dim + 1]);
        if constexpr (kPack == 4) {
            q2 = static_cast<float>(q_ptr[dim + 2]);
            q3 = static_cast<float>(q_ptr[dim + 3]);
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
    __shared__ float warp_sums[TOKENS_PER_TILE][kComputeWarps];
    __shared__ float alpha_shared;
    __shared__ float weights_shared[TOKENS_PER_TILE];

    const int pbs = static_cast<int>(page_block_size);

    static_assert(sizeof(Tdata) == 2, "CTA tile kernel assumes 16B chunks map to 8 elements for fp16/bf16.");
    constexpr int CHUNK_ELEMS = 8; // 8 * 2 bytes = 16 bytes.
    constexpr int CHUNKS = HEAD_SIZE / CHUNK_ELEMS;
    constexpr int LOADS_PER_TILE = CHUNKS * TOKENS_PER_TILE;

    // Multi-stage cp.async pipeline. Using >= 3 stages allows us to keep
    // multiple groups in-flight and overlap global->shared copies with compute.
    constexpr int STAGES = 3;
    __shared__ __align__(16) Tdata sh_k[STAGES][TOKENS_PER_TILE][HEAD_SIZE];
    __shared__ __align__(16) Tdata sh_v[STAGES][TOKENS_PER_TILE][HEAD_SIZE];

    const float alibi_slope = (alibi_slopes_ == nullptr) ? 0.0f : alibi_slopes_[head_idx];
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    int t_base = 0;
    for (int logical_block = 0; t_base < seq_len; ++logical_block, t_base += pbs) {
        const int physical_block = static_cast<int>(block_table[logical_block]);

        const Tdata *k_base = k_cache_ + physical_block * k_batch_stride + kv_head_idx * k_head_stride;
        const Tdata *v_base = v_cache_ + physical_block * v_batch_stride + kv_head_idx * v_head_stride;

        const int token_end = min(pbs, seq_len - t_base);
        const int num_tiles = (token_end + TOKENS_PER_TILE - 1) / TOKENS_PER_TILE;
        if (num_tiles <= 0) {
            continue;
        }

        int pending_groups = 0;
        const int preload = min(STAGES, num_tiles);
        for (int ti = 0; ti < preload; ++ti) {
            const int token_in_block = ti * TOKENS_PER_TILE;
            const int tile_n = min(TOKENS_PER_TILE, token_end - token_in_block);
            for (int li = tid; li < LOADS_PER_TILE; li += CTA_THREADS) {
                const int tok = li / CHUNKS;
                const int chunk = li - tok * CHUNKS;
                const int off = chunk * CHUNK_ELEMS;
                if (tok < tile_n) {
                    const Tdata *k_src = k_base + (token_in_block + tok) * k_row_stride + off;
                    const Tdata *v_src = v_base + (token_in_block + tok) * v_row_stride + off;
                    cpAsyncCaSharedGlobal16(&sh_k[ti][tok][off], k_src);
                    cpAsyncCaSharedGlobal16(&sh_v[ti][tok][off], v_src);
                } else {
                    reinterpret_cast<uint4 *>(&sh_k[ti][tok][off])[0] = make_uint4(0, 0, 0, 0);
                    reinterpret_cast<uint4 *>(&sh_v[ti][tok][off])[0] = make_uint4(0, 0, 0, 0);
                }
            }
            cpAsyncCommit();
            ++pending_groups;
        }

        // Ensure tile 0 is ready. We want to keep up to (STAGES - 1) groups
        // in flight for overlap, but still make forward progress in the tail
        // when we stop issuing new prefetch groups.
        int desired_pending = pending_groups - 1;
        if (desired_pending < 0) {
            desired_pending = 0;
        }
        if (desired_pending > (STAGES - 1)) {
            desired_pending = (STAGES - 1);
        }
        cpAsyncWaitGroupRt(desired_pending);
        pending_groups = desired_pending;
        if constexpr (NUM_WARPS == 1) {
            __syncwarp();
        } else {
            __syncthreads();
        }

        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            const int buf = tile_idx % STAGES;
            const int token_in_block = tile_idx * TOKENS_PER_TILE;
            const int tile_n = min(TOKENS_PER_TILE, token_end - token_in_block);

            float partial[TOKENS_PER_TILE];
#pragma unroll
            for (int j = 0; j < TOKENS_PER_TILE; ++j) {
                if (j < tile_n) {
                    float k0 = 0.0f;
                    float k1 = 0.0f;
                    float k2 = 0.0f;
                    float k3 = 0.0f;
#if defined(__CUDA_ARCH__)
                    if constexpr (std::is_same_v<Tdata, half>) {
                        if constexpr (kPack == 2) {
                            const half2 kh2 = *reinterpret_cast<const half2 *>(&sh_k[buf][j][dim]);
                            const float2 kf = __half22float2(kh2);
                            k0 = kf.x;
                            k1 = kf.y;
                        } else {
                            const half2 kh2_0 = *reinterpret_cast<const half2 *>(&sh_k[buf][j][dim + 0]);
                            const half2 kh2_1 = *reinterpret_cast<const half2 *>(&sh_k[buf][j][dim + 2]);
                            const float2 kf0 = __half22float2(kh2_0);
                            const float2 kf1 = __half22float2(kh2_1);
                            k0 = kf0.x;
                            k1 = kf0.y;
                            k2 = kf1.x;
                            k3 = kf1.y;
                        }
                    } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                        if constexpr (kPack == 2) {
                            const __nv_bfloat162 kb2 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_k[buf][j][dim]);
                            const float2 kf = __bfloat1622float2(kb2);
                            k0 = kf.x;
                            k1 = kf.y;
                        } else {
                            const __nv_bfloat162 kb2_0 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_k[buf][j][dim + 0]);
                            const __nv_bfloat162 kb2_1 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_k[buf][j][dim + 2]);
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
            for (int j = 0; j < TOKENS_PER_TILE; ++j) {
                float sum = warpReduceSum(partial[j]);
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
                // serial loops in lane0. TOKENS_PER_TILE <= 16 by construction.
                float score = -INFINITY;
                if (lane < TOKENS_PER_TILE && lane < tile_n) {
                    float qk = 0.0f;
#pragma unroll
                    for (int w = 0; w < kComputeWarps; ++w) {
                        qk += warp_sums[lane][w];
                    }
                    const int t = t_base + token_in_block + lane;
                    score = qk * scale_log2;
                    if (alibi_slope != 0.0f) {
                        score += (alibi_slope * static_cast<float>(t - (seq_len - 1))) * kLog2e;
                    }
                }

                float tile_max = warpReduceMax(score);
                tile_max = __shfl_sync(0xffffffff, tile_max, 0);

                float m_new = 0.0f;
                if (lane == 0) {
                    m_new = fmaxf(m, tile_max);
                }
                m_new = __shfl_sync(0xffffffff, m_new, 0);

                float w = 0.0f;
                if (lane < TOKENS_PER_TILE && lane < tile_n) {
                    w = exp2f(score - m_new);
                }

                if (lane < TOKENS_PER_TILE) {
                    weights_shared[lane] = (lane < tile_n) ? w : 0.0f;
                }

                float tile_sum = warpReduceSum(w);
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
            for (int j = 0; j < TOKENS_PER_TILE; ++j) {
                const float w = weights_shared[j];
                float v0 = 0.0f;
                float v1 = 0.0f;
                float v2 = 0.0f;
                float v3 = 0.0f;
#if defined(__CUDA_ARCH__)
                if constexpr (std::is_same_v<Tdata, half>) {
                    if constexpr (kPack == 2) {
                        const half2 vh2 = *reinterpret_cast<const half2 *>(&sh_v[buf][j][dim]);
                        const float2 vf = __half22float2(vh2);
                        v0 = vf.x;
                        v1 = vf.y;
                    } else {
                        const half2 vh2_0 = *reinterpret_cast<const half2 *>(&sh_v[buf][j][dim + 0]);
                        const half2 vh2_1 = *reinterpret_cast<const half2 *>(&sh_v[buf][j][dim + 2]);
                        const float2 vf0 = __half22float2(vh2_0);
                        const float2 vf1 = __half22float2(vh2_1);
                        v0 = vf0.x;
                        v1 = vf0.y;
                        v2 = vf1.x;
                        v3 = vf1.y;
                    }
                } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                    if constexpr (kPack == 2) {
                        const __nv_bfloat162 vb2 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_v[buf][j][dim]);
                        const float2 vf = __bfloat1622float2(vb2);
                        v0 = vf.x;
                        v1 = vf.y;
                    } else {
                        const __nv_bfloat162 vb2_0 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_v[buf][j][dim + 0]);
                        const __nv_bfloat162 vb2_1 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_v[buf][j][dim + 2]);
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

            // Prefetch the tile that will reuse this buffer (STAGES steps ahead).
            const int prefetch_tile = tile_idx + STAGES;
            if (prefetch_tile < num_tiles) {
                const int token_prefetch = prefetch_tile * TOKENS_PER_TILE;
                const int prefetch_n = min(TOKENS_PER_TILE, token_end - token_prefetch);
                for (int li = tid; li < LOADS_PER_TILE; li += CTA_THREADS) {
                    const int tok = li / CHUNKS;
                    const int chunk = li - tok * CHUNKS;
                    const int off = chunk * CHUNK_ELEMS;
                    if (tok < prefetch_n) {
                        const Tdata *k_src = k_base + (token_prefetch + tok) * k_row_stride + off;
                        const Tdata *v_src = v_base + (token_prefetch + tok) * v_row_stride + off;
                        cpAsyncCaSharedGlobal16(&sh_k[buf][tok][off], k_src);
                        cpAsyncCaSharedGlobal16(&sh_v[buf][tok][off], v_src);
                    } else {
                        reinterpret_cast<uint4 *>(&sh_k[buf][tok][off])[0] = make_uint4(0, 0, 0, 0);
                        reinterpret_cast<uint4 *>(&sh_v[buf][tok][off])[0] = make_uint4(0, 0, 0, 0);
                    }
                }
                cpAsyncCommit();
                ++pending_groups;
            }

            if (tile_idx + 1 < num_tiles) {
                // Before consuming the next tile, ensure at least one group
                // completes. In steady state we keep (STAGES - 1) in flight; in
                // the tail (no more prefetches) we gradually drain.
                int desired_pending = pending_groups - 1;
                if (desired_pending < 0) {
                    desired_pending = 0;
                }
                if (desired_pending > (STAGES - 1)) {
                    desired_pending = (STAGES - 1);
                }
                cpAsyncWaitGroupRt(desired_pending);
                pending_groups = desired_pending;
                if constexpr (NUM_WARPS == 1) {
                    __syncwarp();
                } else {
                    __syncthreads();
                }
            }
        }

        // Drain any in-flight async copies before moving to the next paged block.
        cpAsyncWaitAll();
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
        if constexpr (std::is_same_v<Tdata, half>) {
            out_ptr[dim + 0] = __float2half_rn(o0);
            out_ptr[dim + 1] = __float2half_rn(o1);
            if constexpr (kPack == 4) {
                out_ptr[dim + 2] = __float2half_rn(o2);
                out_ptr[dim + 3] = __float2half_rn(o3);
            }
        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
            out_ptr[dim + 0] = __float2bfloat16_rn(o0);
            out_ptr[dim + 1] = __float2bfloat16_rn(o1);
            if constexpr (kPack == 4) {
                out_ptr[dim + 2] = __float2bfloat16_rn(o2);
                out_ptr[dim + 3] = __float2bfloat16_rn(o3);
            }
        } else
#endif
        {
            out_ptr[dim + 0] = static_cast<Tdata>(o0);
            out_ptr[dim + 1] = static_cast<Tdata>(o1);
            if constexpr (kPack == 4) {
                out_ptr[dim + 2] = static_cast<Tdata>(o2);
                out_ptr[dim + 3] = static_cast<Tdata>(o3);
            }
        }
    }
}

// GQA/MQA fused decode kernel: one CTA computes outputs for NGROUPS query heads that
// share the same KV head. This reduces redundant K/V reads when num_heads > num_kv_heads.
//
// v0.4: implemented for head_dim=128 and NGROUPS=4 (common case: 32 Q heads / 8 KV heads).
template <typename Tindex, typename Tdata, int HEAD_SIZE, int CTA_THREADS, int TOKENS_PER_TILE, int NGROUPS>
__device__ void flashAttentionDecodeCtaGqaKernel(
    Tdata *out_,
    const Tdata *q_,
    const Tdata *k_cache_,
    const Tdata *v_cache_,
    const Tindex *block_tables_,
    const Tindex *cache_lens_,
    const float *alibi_slopes_,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    ptrdiff_t q_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t o_stride) {

    constexpr int kWarpSize = 32;
    static_assert(HEAD_SIZE == 128, "v0.4 GQA fused CTA kernel is implemented for head_size=128 only.");
    static_assert(NGROUPS == 4, "v0.4 GQA fused CTA kernel is implemented for NGROUPS=4 only.");
    static_assert(CTA_THREADS % kWarpSize == 0, "CTA_THREADS must be a multiple of 32.");
    static_assert(TOKENS_PER_TILE > 0 && TOKENS_PER_TILE <= 16, "TOKENS_PER_TILE should stay small.");
    constexpr int NUM_WARPS = CTA_THREADS / kWarpSize;

    // Pack dims per thread. For head_dim=128 and CTA_THREADS=64, kPack=2.
    static_assert(HEAD_SIZE % CTA_THREADS == 0, "HEAD_SIZE must be divisible by CTA_THREADS.");
    constexpr int kPack = HEAD_SIZE / CTA_THREADS;
    static_assert(kPack == 2, "v0.4 GQA fused CTA kernel expects kPack=2.");
    constexpr int kPackedDims = CTA_THREADS;
    constexpr int kComputeWarps = (kPackedDims + kWarpSize - 1) / kWarpSize;

    const int seq_idx = blockIdx.y;
    const int kv_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % kWarpSize;
    const int warp_id = tid / kWarpSize;
    const int dim = tid * kPack;

    const int seq_len = static_cast<int>(cache_lens_[seq_idx]);
    if (seq_len <= 0) {
        return;
    }

    // v0.4 limitation: alibi slopes are per query head; support can be added later.
    if (alibi_slopes_ != nullptr) {
        return;
    }

    const Tindex *block_table = block_tables_ + seq_idx * static_cast<int>(max_num_blocks_per_seq);

    // q/out are [num_seqs, num_heads, head_size]. For a KV head, we handle NGROUPS query heads:
    // q_head = kv_head * NGROUPS + g
    float q0[NGROUPS];
    float q1[NGROUPS];
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_same_v<Tdata, half>) {
#pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int q_head = kv_head_idx * NGROUPS + g;
            const Tdata *q_ptr = q_ + seq_idx * q_stride + q_head * HEAD_SIZE;
            const half2 qh2 = *reinterpret_cast<const half2 *>(q_ptr + dim);
            const float2 qf = __half22float2(qh2);
            q0[g] = qf.x;
            q1[g] = qf.y;
        }
    } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
#pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int q_head = kv_head_idx * NGROUPS + g;
            const Tdata *q_ptr = q_ + seq_idx * q_stride + q_head * HEAD_SIZE;
            const __nv_bfloat162 qb2 = *reinterpret_cast<const __nv_bfloat162 *>(q_ptr + dim);
            const float2 qf = __bfloat1622float2(qb2);
            q0[g] = qf.x;
            q1[g] = qf.y;
        }
    } else
#endif
    {
#pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int q_head = kv_head_idx * NGROUPS + g;
            const Tdata *q_ptr = q_ + seq_idx * q_stride + q_head * HEAD_SIZE;
            q0[g] = static_cast<float>(q_ptr[dim + 0]);
            q1[g] = static_cast<float>(q_ptr[dim + 1]);
        }
    }

    float acc0[NGROUPS];
    float acc1[NGROUPS];
    float m[NGROUPS];
    float l[NGROUPS];
#pragma unroll
    for (int g = 0; g < NGROUPS; ++g) {
        acc0[g] = 0.0f;
        acc1[g] = 0.0f;
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }

    __shared__ float warp_sums[NGROUPS][TOKENS_PER_TILE][kComputeWarps];
    __shared__ float alpha_shared[NGROUPS];
    __shared__ float weights_shared[NGROUPS][TOKENS_PER_TILE];

    const int pbs = static_cast<int>(page_block_size);
    constexpr float kLog2e = 1.4426950408889634f;
    const float scale_log2 = scale * kLog2e;

    static_assert(sizeof(Tdata) == 2, "CTA GQA kernel assumes fp16/bf16.");
    constexpr int CHUNK_ELEMS = 8; // 8 * 2 bytes = 16 bytes.
    constexpr int CHUNKS = HEAD_SIZE / CHUNK_ELEMS;
    constexpr int LOADS_PER_TILE = CHUNKS * TOKENS_PER_TILE;

    constexpr int STAGES = 3;
    __shared__ __align__(16) Tdata sh_k[STAGES][TOKENS_PER_TILE][HEAD_SIZE];
    __shared__ __align__(16) Tdata sh_v[STAGES][TOKENS_PER_TILE][HEAD_SIZE];

    int t_base = 0;
    for (int logical_block = 0; t_base < seq_len; ++logical_block, t_base += pbs) {
        const int physical_block = static_cast<int>(block_table[logical_block]);

        const Tdata *k_base = k_cache_ + physical_block * k_batch_stride + kv_head_idx * k_head_stride;
        const Tdata *v_base = v_cache_ + physical_block * v_batch_stride + kv_head_idx * v_head_stride;

        const int token_end = min(pbs, seq_len - t_base);
        const int num_tiles = (token_end + TOKENS_PER_TILE - 1) / TOKENS_PER_TILE;
        if (num_tiles <= 0) {
            continue;
        }

        int pending_groups = 0;
        const int preload = min(STAGES, num_tiles);
        for (int ti = 0; ti < preload; ++ti) {
            const int token_in_block = ti * TOKENS_PER_TILE;
            const int tile_n = min(TOKENS_PER_TILE, token_end - token_in_block);
            for (int li = tid; li < LOADS_PER_TILE; li += CTA_THREADS) {
                const int tok = li / CHUNKS;
                const int chunk = li - tok * CHUNKS;
                const int off = chunk * CHUNK_ELEMS;
                if (tok < tile_n) {
                    const Tdata *k_src = k_base + (token_in_block + tok) * k_row_stride + off;
                    const Tdata *v_src = v_base + (token_in_block + tok) * v_row_stride + off;
                    cpAsyncCaSharedGlobal16(&sh_k[ti][tok][off], k_src);
                    cpAsyncCaSharedGlobal16(&sh_v[ti][tok][off], v_src);
                } else {
                    reinterpret_cast<uint4 *>(&sh_k[ti][tok][off])[0] = make_uint4(0, 0, 0, 0);
                    reinterpret_cast<uint4 *>(&sh_v[ti][tok][off])[0] = make_uint4(0, 0, 0, 0);
                }
            }
            cpAsyncCommit();
            ++pending_groups;
        }

        int desired_pending = pending_groups - 1;
        if (desired_pending < 0) {
            desired_pending = 0;
        }
        if (desired_pending > (STAGES - 1)) {
            desired_pending = (STAGES - 1);
        }
        cpAsyncWaitGroupRt(desired_pending);
        pending_groups = desired_pending;
        if constexpr (NUM_WARPS == 1) {
            __syncwarp();
        } else {
            __syncthreads();
        }

        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            const int buf = tile_idx % STAGES;
            const int token_in_block = tile_idx * TOKENS_PER_TILE;
            const int tile_n = min(TOKENS_PER_TILE, token_end - token_in_block);

            // Compute QK partial sums for each group and each token in the tile.
            float partial_qk[NGROUPS][TOKENS_PER_TILE];
#pragma unroll
            for (int g = 0; g < NGROUPS; ++g) {
#pragma unroll
                for (int j = 0; j < TOKENS_PER_TILE; ++j) {
                    if (j < tile_n) {
                        float k0 = 0.0f;
                        float k1 = 0.0f;
#if defined(__CUDA_ARCH__)
                        if constexpr (std::is_same_v<Tdata, half>) {
                            const half2 kh2 = *reinterpret_cast<const half2 *>(&sh_k[buf][j][dim]);
                            const float2 kf = __half22float2(kh2);
                            k0 = kf.x;
                            k1 = kf.y;
                        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                            const __nv_bfloat162 kb2 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_k[buf][j][dim]);
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
            for (int g = 0; g < NGROUPS; ++g) {
#pragma unroll
                for (int j = 0; j < TOKENS_PER_TILE; ++j) {
                    const float sum = warpReduceSum(partial_qk[g][j]);
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
                for (int g = 0; g < NGROUPS; ++g) {
                    float score = -INFINITY;
                    if (lane < TOKENS_PER_TILE && lane < tile_n) {
                        float qk = 0.0f;
#pragma unroll
                        for (int w = 0; w < kComputeWarps; ++w) {
                            qk += warp_sums[g][lane][w];
                        }
                        score = qk * scale_log2;
                    }

                    float tile_max = warpReduceMax(score);
                    tile_max = __shfl_sync(0xffffffff, tile_max, 0);

                    float m_new = 0.0f;
                    if (lane == 0) {
                        m_new = fmaxf(m[g], tile_max);
                    }
                    m_new = __shfl_sync(0xffffffff, m_new, 0);

                    float w = 0.0f;
                    if (lane < TOKENS_PER_TILE && lane < tile_n) {
                        w = exp2f(score - m_new);
                    }
                    if (lane < TOKENS_PER_TILE) {
                        weights_shared[g][lane] = (lane < tile_n) ? w : 0.0f;
                    }

                    const float tile_sum = warpReduceSum(w);
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

            float alpha[NGROUPS];
            float sum_wv0[NGROUPS];
            float sum_wv1[NGROUPS];
#pragma unroll
            for (int g = 0; g < NGROUPS; ++g) {
                alpha[g] = alpha_shared[g];
                sum_wv0[g] = 0.0f;
                sum_wv1[g] = 0.0f;
            }

#pragma unroll
            for (int j = 0; j < TOKENS_PER_TILE; ++j) {
                float v0 = 0.0f;
                float v1 = 0.0f;
#if defined(__CUDA_ARCH__)
                if constexpr (std::is_same_v<Tdata, half>) {
                    const half2 vh2 = *reinterpret_cast<const half2 *>(&sh_v[buf][j][dim]);
                    const float2 vf = __half22float2(vh2);
                    v0 = vf.x;
                    v1 = vf.y;
                } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
                    const __nv_bfloat162 vb2 = *reinterpret_cast<const __nv_bfloat162 *>(&sh_v[buf][j][dim]);
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
                for (int g = 0; g < NGROUPS; ++g) {
                    const float w = weights_shared[g][j];
                    sum_wv0[g] = fmaf(w, v0, sum_wv0[g]);
                    sum_wv1[g] = fmaf(w, v1, sum_wv1[g]);
                }
            }

#pragma unroll
            for (int g = 0; g < NGROUPS; ++g) {
                acc0[g] = acc0[g] * alpha[g] + sum_wv0[g];
                acc1[g] = acc1[g] * alpha[g] + sum_wv1[g];
            }

            const int prefetch_tile = tile_idx + STAGES;
            if (prefetch_tile < num_tiles) {
                const int token_prefetch = prefetch_tile * TOKENS_PER_TILE;
                const int prefetch_n = min(TOKENS_PER_TILE, token_end - token_prefetch);
                for (int li = tid; li < LOADS_PER_TILE; li += CTA_THREADS) {
                    const int tok = li / CHUNKS;
                    const int chunk = li - tok * CHUNKS;
                    const int off = chunk * CHUNK_ELEMS;
                    if (tok < prefetch_n) {
                        const Tdata *k_src = k_base + (token_prefetch + tok) * k_row_stride + off;
                        const Tdata *v_src = v_base + (token_prefetch + tok) * v_row_stride + off;
                        cpAsyncCaSharedGlobal16(&sh_k[buf][tok][off], k_src);
                        cpAsyncCaSharedGlobal16(&sh_v[buf][tok][off], v_src);
                    } else {
                        reinterpret_cast<uint4 *>(&sh_k[buf][tok][off])[0] = make_uint4(0, 0, 0, 0);
                        reinterpret_cast<uint4 *>(&sh_v[buf][tok][off])[0] = make_uint4(0, 0, 0, 0);
                    }
                }
                cpAsyncCommit();
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
                cpAsyncWaitGroupRt(desired_pending2);
                pending_groups = desired_pending2;
                if constexpr (NUM_WARPS == 1) {
                    __syncwarp();
                } else {
                    __syncthreads();
                }
            }
        }

        cpAsyncWaitAll();
        if constexpr (NUM_WARPS == 1) {
            __syncwarp();
        } else {
            __syncthreads();
        }
    }

    // Write outputs for each group.
    __shared__ float inv_l_shared[NGROUPS];
    if (tid < NGROUPS) {
        inv_l_shared[tid] = 1.0f / (l[tid] + 1e-6f);
    }
    if constexpr (NUM_WARPS == 1) {
        __syncwarp();
    } else {
        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < NGROUPS; ++g) {
        const int q_head = kv_head_idx * NGROUPS + g;
        Tdata *out_ptr = out_ + seq_idx * o_stride + q_head * HEAD_SIZE;
        const float s = inv_l_shared[g];
        const float o0 = acc0[g] * s;
        const float o1 = acc1[g] * s;
#if defined(__CUDA_ARCH__)
        if constexpr (std::is_same_v<Tdata, half>) {
            out_ptr[dim + 0] = __float2half_rn(o0);
            out_ptr[dim + 1] = __float2half_rn(o1);
        } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
            out_ptr[dim + 0] = __float2bfloat16_rn(o0);
            out_ptr[dim + 1] = __float2bfloat16_rn(o1);
        } else
#endif
        {
            out_ptr[dim + 0] = static_cast<Tdata>(o0);
            out_ptr[dim + 1] = static_cast<Tdata>(o1);
        }
    }
}
} // namespace op::paged_attention::cuda

#endif // __PAGED_ATTENTION_KERNEL_V2_CUH__
