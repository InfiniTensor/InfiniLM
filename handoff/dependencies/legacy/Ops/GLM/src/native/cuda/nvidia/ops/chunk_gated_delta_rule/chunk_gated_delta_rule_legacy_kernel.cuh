#pragma once
#include <cstdint>
// Exact baseline helpers and active state-workspace kernel. The baseline
// chunked-matrix branch is hard-disabled and is not copied as a second path.
namespace infini::ops::chunk_native_detail {
__device__ inline int64_t loadOptionalIndex(
    const void *indices,
    bool is_i64,
    int idx,
    int fallback) {
    if (indices == nullptr) {
        return static_cast<int64_t>(fallback);
    }
    return is_i64
             ? static_cast<const int64_t *>(indices)[idx]
             : static_cast<int64_t>(static_cast<const int32_t *>(indices)[idx]);
}

template <typename T>
__device__ inline float loadAsFloat(const T *ptr, ptrdiff_t offset) {
    return static_cast<float>(ptr[offset]);
}

template <>
__device__ inline float loadAsFloat<half>(const half *ptr, ptrdiff_t offset) {
    return __half2float(ptr[offset]);
}

template <>
__device__ inline float loadAsFloat<cuda_bfloat16>(const cuda_bfloat16 *ptr, ptrdiff_t offset) {
    return __bfloat162float(ptr[offset]);
}

#define CGDR_FOR(idx, n) \
    for (int idx = threadIdx.x; idx < static_cast<int>(n); idx += blockDim.x)

template <typename Tcompute, int NUM_THREADS>
__device__ Tcompute blockReduceSum(Tcompute v) {
    __shared__ Tcompute smem[NUM_THREADS];
    smem[threadIdx.x] = v;
    __syncthreads();

    for (int s = NUM_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    return smem[0];
}

template <
    typename Tdata,
    typename Tgate,
    typename Tcompute,
    size_t Dk,
    size_t Dv,
    size_t NUM_THREADS>
__device__ void chunkGatedDeltaRuleRecurrentKernel(
    Tcompute *state_workspace,
    Tdata *out,
    Tdata *initial_state,
    Tdata *final_state,
    const Tdata *q,
    const Tdata *k,
    const Tdata *v,
    const Tgate *g,
    const Tgate *beta,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool cu_seqlens_i64,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    bool use_qk_l2norm,
    bool has_cu_seqlens,
    bool indexed_state_pool,
    size_t T,
    size_t,
    size_t pool_size,
    size_t Hk,
    size_t value_heads_per_key_head,
    ptrdiff_t out_s0,
    ptrdiff_t out_s1,
    ptrdiff_t out_s2,
    ptrdiff_t initial_s0,
    ptrdiff_t initial_s1,
    ptrdiff_t initial_s2,
    ptrdiff_t initial_s3,
    ptrdiff_t final_s0,
    ptrdiff_t final_s1,
    ptrdiff_t final_s2,
    ptrdiff_t final_s3,
    ptrdiff_t q_s0,
    ptrdiff_t q_s1,
    ptrdiff_t q_s2,
    ptrdiff_t k_s0,
    ptrdiff_t k_s1,
    ptrdiff_t k_s2,
    ptrdiff_t v_s0,
    ptrdiff_t v_s1,
    ptrdiff_t v_s2,
    ptrdiff_t g_s0,
    ptrdiff_t g_s1,
    ptrdiff_t g_s2,
    ptrdiff_t beta_s0,
    ptrdiff_t beta_s1,
    ptrdiff_t beta_s2) {

    const int batch_idx = blockIdx.x;
    const int value_head_idx = blockIdx.y;
    const int key_head_idx = value_head_idx / static_cast<int>(value_heads_per_key_head);

    if (key_head_idx >= static_cast<int>(Hk)) {
        return;
    }

    int64_t token_begin = 0;
    int64_t token_end = static_cast<int64_t>(T);

    if (has_cu_seqlens) {
        token_begin = loadOptionalIndex(cu_seqlens, cu_seqlens_i64, batch_idx, 0);
        token_end = loadOptionalIndex(cu_seqlens, cu_seqlens_i64, batch_idx + 1, 0);
        if (token_begin < 0 || token_end < token_begin || token_end > static_cast<int64_t>(T)) {
            return;
        }
    }

    int64_t read_slot = batch_idx;
    int64_t write_slot = batch_idx;

    if (indexed_state_pool) {
        read_slot = loadOptionalIndex(
            initial_state_indices,
            initial_state_indices_i64,
            batch_idx,
            batch_idx);

        write_slot = final_state_indices == nullptr
                       ? static_cast<int64_t>(batch_idx)
                       : loadOptionalIndex(
                           final_state_indices,
                           final_state_indices_i64,
                           batch_idx,
                           batch_idx);

        if (read_slot < 0 || write_slot < 0 || read_slot >= static_cast<int64_t>(pool_size) || write_slot >= static_cast<int64_t>(pool_size)) {
            return;
        }
    }

    const ptrdiff_t initial_base = indexed_state_pool
                                     ? static_cast<ptrdiff_t>(read_slot) * initial_s0 + static_cast<ptrdiff_t>(value_head_idx) * initial_s1
                                     : static_cast<ptrdiff_t>(batch_idx) * initial_s0 + static_cast<ptrdiff_t>(value_head_idx) * initial_s1;

    Tdata *final_state_target = nullptr;
    ptrdiff_t final_base = 0;

    if (indexed_state_pool && final_state_indices != nullptr) {
        final_state_target = initial_state;
        final_base = static_cast<ptrdiff_t>(write_slot) * initial_s0 + static_cast<ptrdiff_t>(value_head_idx) * initial_s1;
    } else {
        final_state_target = final_state;
        final_base = static_cast<ptrdiff_t>(batch_idx) * final_s0 + static_cast<ptrdiff_t>(value_head_idx) * final_s1;
    }

    const ptrdiff_t workspace_block = (static_cast<ptrdiff_t>(batch_idx) * gridDim.y + static_cast<ptrdiff_t>(value_head_idx)) * static_cast<ptrdiff_t>(Dk * Dv);

    Tcompute *state_local = state_workspace + workspace_block;
    __shared__ Tcompute q_vec[Dk];
    __shared__ Tcompute k_vec[Dk];
    __shared__ Tcompute v_new[Dv];
    __shared__ Tcompute scalar_buf[3];

    const int token_batch = has_cu_seqlens ? 0 : batch_idx;
    const Tcompute scale = rsqrtf(static_cast<Tcompute>(Dk));

    CGDR_FOR(i, Dk * Dv) {
        int dk = i / Dv;
        int dv = i % Dv;

        ptrdiff_t read_idx = initial_base + static_cast<ptrdiff_t>(dv) * initial_s2 + static_cast<ptrdiff_t>(dk) * initial_s3;

        state_local[i] = static_cast<Tcompute>(
            loadAsFloat(initial_state, read_idx));
    }
    __syncthreads();

    for (int64_t token_idx = token_begin; token_idx < token_end; ++token_idx) {
        ptrdiff_t q_base = static_cast<ptrdiff_t>(token_batch) * q_s0 + static_cast<ptrdiff_t>(token_idx) * q_s1 + static_cast<ptrdiff_t>(key_head_idx) * q_s2;
        ptrdiff_t k_base = static_cast<ptrdiff_t>(token_batch) * k_s0 + static_cast<ptrdiff_t>(token_idx) * k_s1 + static_cast<ptrdiff_t>(key_head_idx) * k_s2;

        Tcompute q_sum = 0;
        Tcompute k_sum = 0;
        for (int dk = threadIdx.x; dk < static_cast<int>(Dk); dk += blockDim.x) {
            Tcompute q_raw = static_cast<Tcompute>(loadAsFloat(q, q_base + dk));
            Tcompute k_raw = static_cast<Tcompute>(loadAsFloat(k, k_base + dk));
            q_vec[dk] = q_raw;
            k_vec[dk] = k_raw;
            q_sum += q_raw * q_raw;
            k_sum += k_raw * k_raw;
        }

        q_sum = blockReduceSum<Tcompute, NUM_THREADS>(q_sum);
        k_sum = blockReduceSum<Tcompute, NUM_THREADS>(k_sum);

        if (threadIdx.x == 0) {
            ptrdiff_t gate_offset = static_cast<ptrdiff_t>(token_batch) * g_s0 + static_cast<ptrdiff_t>(token_idx) * g_s1 + static_cast<ptrdiff_t>(value_head_idx) * g_s2;
            ptrdiff_t beta_offset = static_cast<ptrdiff_t>(token_batch) * beta_s0 + static_cast<ptrdiff_t>(token_idx) * beta_s1 + static_cast<ptrdiff_t>(value_head_idx) * beta_s2;

            scalar_buf[0] = expf(static_cast<Tcompute>(loadAsFloat(g, gate_offset)));
            scalar_buf[1] = static_cast<Tcompute>(loadAsFloat(beta, beta_offset));
            scalar_buf[2] = use_qk_l2norm
                              ? rsqrtf(q_sum + static_cast<Tcompute>(1e-6)) * scale
                              : scale;
        }
        __syncthreads();

        const Tcompute decay = scalar_buf[0];
        const Tcompute beta_t = scalar_buf[1];
        const Tcompute q_scale = scalar_buf[2];
        const Tcompute k_scale = use_qk_l2norm
                                   ? rsqrtf(k_sum + static_cast<Tcompute>(1e-6))
                                   : static_cast<Tcompute>(1);

        for (int dk = threadIdx.x; dk < static_cast<int>(Dk); dk += blockDim.x) {
            q_vec[dk] *= q_scale;
            k_vec[dk] *= k_scale;
        }
        __syncthreads();

        for (int dv = threadIdx.x; dv < static_cast<int>(Dv); dv += blockDim.x) {
            Tcompute projected = 0;
            for (int dk = 0; dk < static_cast<int>(Dk); ++dk) {
                projected += k_vec[dk] * state_local[dk * Dv + dv];
            }

            ptrdiff_t v_base = static_cast<ptrdiff_t>(token_batch) * v_s0 + static_cast<ptrdiff_t>(token_idx) * v_s1 + static_cast<ptrdiff_t>(value_head_idx) * v_s2;
            Tcompute v_raw = static_cast<Tcompute>(loadAsFloat(v, v_base + dv));
            v_new[dv] = beta_t * (v_raw - decay * projected);
        }
        __syncthreads();

        CGDR_FOR(x, Dk * Dv) {
            int dk = x / Dv;
            int dv = x % Dv;
            state_local[x] = decay * state_local[x] + k_vec[dk] * v_new[dv];
        }
        __syncthreads();

        for (int dv = threadIdx.x; dv < static_cast<int>(Dv); dv += blockDim.x) {
            Tcompute out_val = 0;
            for (int dk = 0; dk < static_cast<int>(Dk); ++dk) {
                out_val += q_vec[dk] * state_local[dk * Dv + dv];
            }

            ptrdiff_t out_base = static_cast<ptrdiff_t>(token_batch) * out_s0 + static_cast<ptrdiff_t>(token_idx) * out_s1 + static_cast<ptrdiff_t>(value_head_idx) * out_s2;
            out[out_base + dv] = static_cast<Tdata>(out_val);
        }
        __syncthreads();
    }

    CGDR_FOR(i, Dk * Dv) {
        int dk = i / Dv;
        int dv = i % Dv;

        const ptrdiff_t s2 = final_state_indices != nullptr ? initial_s2 : final_s2;
        const ptrdiff_t s3 = final_state_indices != nullptr ? initial_s3 : final_s3;
        ptrdiff_t write_idx = final_base + static_cast<ptrdiff_t>(dv) * s2 + static_cast<ptrdiff_t>(dk) * s3;

        final_state_target[write_idx] = static_cast<Tdata>(state_local[i]);
    }
}
#undef CGDR_FOR
}
