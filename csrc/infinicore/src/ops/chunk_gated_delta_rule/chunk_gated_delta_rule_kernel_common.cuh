#ifndef __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__
#define __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__

#include <cstddef>
#include <cstdint>
#include <maca_fp16.h>
#include <maca_bfloat16.h>

using half = __half;
using cuda_bfloat16 = __maca_bfloat16;
#ifndef __nv_bfloat16
using __nv_bfloat16 = __maca_bfloat16;
#endif


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
    const Tcompute total = smem[0];
    // Both Q and K reductions reuse this shared array. All warps must read
    // the result before the next invocation can overwrite it.
    __syncthreads();
    return total;
}

template <
    typename Tdata,
    typename Tgate,
    typename Tcompute,
    size_t Dk,
    size_t Dv,
    size_t NUM_THREADS>
__device__ void chunkGatedDeltaRuleKernel(
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
    size_t chunk_size,
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

    const ptrdiff_t per_block_workspace = static_cast<ptrdiff_t>(Dk * Dv) + static_cast<ptrdiff_t>(chunk_size * Dk) * 3 + static_cast<ptrdiff_t>(chunk_size * Dv) * 3 + static_cast<ptrdiff_t>(chunk_size * chunk_size) + static_cast<ptrdiff_t>(chunk_size) * 3;

    const ptrdiff_t workspace_block = (static_cast<ptrdiff_t>(batch_idx) * gridDim.y + static_cast<ptrdiff_t>(value_head_idx)) * per_block_workspace;

    Tcompute *state_local = state_workspace + workspace_block;
    Tcompute *q_buf = state_local + Dk * Dv;
    Tcompute *k_buf = q_buf + chunk_size * Dk;
    Tcompute *k_cumdecay = k_buf + chunk_size * Dk;
    Tcompute *v_beta = k_cumdecay + chunk_size * Dk;
    Tcompute *v_mid = v_beta + chunk_size * Dv;
    Tcompute *v_new = v_mid + chunk_size * Dv;
    Tcompute *attn = v_new + chunk_size * Dv;
    Tcompute *g_cum = attn + chunk_size * chunk_size;
    Tcompute *beta_buf = g_cum + chunk_size;
    Tcompute *row_buf = beta_buf + chunk_size;

    const int token_batch = has_cu_seqlens ? 0 : batch_idx;
    const Tcompute scale = rsqrtf(static_cast<Tcompute>(Dk));

    // Load initial state.
    CGDR_FOR(i, Dk * Dv) {
        int dk = i / Dv;
        int dv = i % Dv;

        ptrdiff_t read_idx = initial_base + static_cast<ptrdiff_t>(dv) * initial_s2 + static_cast<ptrdiff_t>(dk) * initial_s3;

        state_local[i] = static_cast<Tcompute>(
            loadAsFloat(initial_state, read_idx));
    }
    __syncthreads();

    for (int64_t chunk_begin = token_begin;
         chunk_begin < token_end;
         chunk_begin += static_cast<int64_t>(chunk_size)) {

        const int64_t remaining = token_end - chunk_begin;
        const int actual_len = static_cast<int>(
            remaining < static_cast<int64_t>(chunk_size)
                ? remaining
                : static_cast<int64_t>(chunk_size));

        // Load beta and cumulative gate.
        if (threadIdx.x == 0) {
            Tcompute running_g = 0;
            for (int t = 0; t < static_cast<int>(chunk_size); ++t) {
                if (t < actual_len) {
                    int64_t token_idx = chunk_begin + t;
                    ptrdiff_t gate_offset = static_cast<ptrdiff_t>(token_batch) * g_s0 + static_cast<ptrdiff_t>(token_idx) * g_s1 + static_cast<ptrdiff_t>(value_head_idx) * g_s2;
                    ptrdiff_t beta_offset = static_cast<ptrdiff_t>(token_batch) * beta_s0 + static_cast<ptrdiff_t>(token_idx) * beta_s1 + static_cast<ptrdiff_t>(value_head_idx) * beta_s2;

                    running_g += static_cast<Tcompute>(
                        loadAsFloat(g, gate_offset));
                    beta_buf[t] = static_cast<Tcompute>(
                        loadAsFloat(beta, beta_offset));
                    g_cum[t] = running_g;
                } else {
                    beta_buf[t] = 0;
                    g_cum[t] = running_g;
                }
            }
        }
        __syncthreads();

        // Load q/k/v_beta.
        CGDR_FOR(x, chunk_size * Dk) {
            int t = x / Dk;
            int dk = x % Dk;

            if (t < actual_len) {
                int64_t token_idx = chunk_begin + t;
                ptrdiff_t q_base = static_cast<ptrdiff_t>(token_batch) * q_s0 + static_cast<ptrdiff_t>(token_idx) * q_s1 + static_cast<ptrdiff_t>(key_head_idx) * q_s2;
                ptrdiff_t k_base = static_cast<ptrdiff_t>(token_batch) * k_s0 + static_cast<ptrdiff_t>(token_idx) * k_s1 + static_cast<ptrdiff_t>(key_head_idx) * k_s2;

                q_buf[x] = static_cast<Tcompute>(loadAsFloat(q, q_base + dk)) * scale;
                k_buf[x] = static_cast<Tcompute>(loadAsFloat(k, k_base + dk));
            } else {
                q_buf[x] = 0;
                k_buf[x] = 0;
            }
        }

        CGDR_FOR(x, chunk_size * Dv) {
            int t = x / Dv;
            int dv = x % Dv;

            if (t < actual_len) {
                int64_t token_idx = chunk_begin + t;
                ptrdiff_t v_base = static_cast<ptrdiff_t>(token_batch) * v_s0 + static_cast<ptrdiff_t>(token_idx) * v_s1 + static_cast<ptrdiff_t>(value_head_idx) * v_s2;

                v_beta[x] = static_cast<Tcompute>(loadAsFloat(v, v_base + dv)) * beta_buf[t];
            } else {
                v_beta[x] = 0;
            }
        }
        __syncthreads();

        // Optional q/k L2 norm.
        if (use_qk_l2norm) {
            for (int t = 0; t < static_cast<int>(chunk_size); ++t) {
                Tcompute q_sum = 0;
                Tcompute k_sum = 0;

                for (int dk = threadIdx.x; dk < static_cast<int>(Dk); dk += blockDim.x) {
                    q_sum += q_buf[t * Dk + dk] * q_buf[t * Dk + dk];
                    k_sum += k_buf[t * Dk + dk] * k_buf[t * Dk + dk];
                }

                q_sum = blockReduceSum<Tcompute, NUM_THREADS>(q_sum);
                k_sum = blockReduceSum<Tcompute, NUM_THREADS>(k_sum);

                Tcompute q_norm = rsqrtf(q_sum / (scale * scale) + 1e-6f);
                Tcompute k_norm = rsqrtf(k_sum + 1e-6f);

                for (int dk = threadIdx.x; dk < static_cast<int>(Dk); dk += blockDim.x) {
                    q_buf[t * Dk + dk] *= q_norm;
                    k_buf[t * Dk + dk] *= k_norm;
                }
                __syncthreads();
            }
        }

        // Build lower-triangular attn.
        CGDR_FOR(x, chunk_size * chunk_size) {
            int i = x / chunk_size;
            int j = x % chunk_size;

            if (j < i) {
                Tcompute dot = 0;
                for (int dk = 0; dk < static_cast<int>(Dk); ++dk) {
                    dot += k_buf[i * Dk + dk] * beta_buf[i] * k_buf[j * Dk + dk];
                }
                attn[x] = -dot * expf(g_cum[i] - g_cum[j]);
            } else {
                attn[x] = 0;
            }
        }
        __syncthreads();

        // Triangular solve-like correction.
        // Sequential in i, parallel in j.
        for (int i = 1; i < static_cast<int>(chunk_size); ++i) {
            CGDR_FOR(m, chunk_size) {
                row_buf[m] = m < i ? attn[i * chunk_size + m] : 0;
            }
            __syncthreads();

            for (int j = threadIdx.x; j < i; j += blockDim.x) {
                Tcompute correction = 0;
                for (int m = 0; m < i; ++m) {
                    correction += row_buf[m] * attn[m * chunk_size + j];
                }
                attn[i * chunk_size + j] = row_buf[j] + correction;
            }
            __syncthreads();
        }

        CGDR_FOR(i, chunk_size) {
            attn[i * chunk_size + i] = 1;
        }
        __syncthreads();

        // v_mid = attn @ v_beta.
        CGDR_FOR(x, chunk_size * Dv) {
            int i = x / Dv;
            int dv = x % Dv;

            Tcompute sum = 0;
            for (int j = 0; j < static_cast<int>(chunk_size); ++j) {
                sum += attn[i * chunk_size + j] * v_beta[j * Dv + dv];
            }
            v_mid[x] = sum;
        }

        // k_cumdecay = attn @ (k * beta * exp(g)).
        CGDR_FOR(x, chunk_size * Dk) {
            int i = x / Dk;
            int dk = x % Dk;

            Tcompute sum = 0;
            for (int j = 0; j < static_cast<int>(chunk_size); ++j) {
                sum += attn[i * chunk_size + j] * k_buf[j * Dk + dk] * beta_buf[j] * expf(g_cum[j]);
            }
            k_cumdecay[x] = sum;
        }
        __syncthreads();

        // v_new = v_mid - k_cumdecay @ state.
        CGDR_FOR(x, chunk_size * Dv) {
            int i = x / Dv;
            int dv = x % Dv;

            Tcompute v_prime = 0;
            for (int dk = 0; dk < static_cast<int>(Dk); ++dk) {
                v_prime += k_cumdecay[i * Dk + dk] * state_local[dk * Dv + dv];
            }
            v_new[x] = v_mid[x] - v_prime;
        }
        __syncthreads();

        // Output.
        CGDR_FOR(x, actual_len * Dv) {
            int i = x / Dv;
            int dv = x % Dv;

            int64_t token_idx = chunk_begin + i;
            ptrdiff_t out_base = static_cast<ptrdiff_t>(token_batch) * out_s0 + static_cast<ptrdiff_t>(token_idx) * out_s1 + static_cast<ptrdiff_t>(value_head_idx) * out_s2;

            Tcompute out_val = 0;
            Tcompute q_decay = expf(g_cum[i]);

            for (int dk = 0; dk < static_cast<int>(Dk); ++dk) {
                out_val += q_buf[i * Dk + dk] * q_decay * state_local[dk * Dv + dv];
            }

            for (int j = 0; j <= i; ++j) {
                Tcompute qk_attn = 0;
                for (int dk = 0; dk < static_cast<int>(Dk); ++dk) {
                    qk_attn += q_buf[i * Dk + dk] * k_buf[j * Dk + dk];
                }

                qk_attn *= expf(g_cum[i] - g_cum[j]);
                out_val += qk_attn * v_new[j * Dv + dv];
            }

            out[out_base + dv] = static_cast<Tdata>(out_val);
        }
        __syncthreads();

        // Update state.
        const Tcompute last_decay = expf(g_cum[chunk_size - 1]);

        CGDR_FOR(x, Dk * Dv) {
            int dk = x / Dv;
            int dv = x % Dv;

            Tcompute next_state = state_local[x] * last_decay;

            for (int i = 0; i < static_cast<int>(chunk_size); ++i) {
                next_state += k_buf[i * Dk + dk] * expf(g_cum[chunk_size - 1] - g_cum[i]) * v_new[i * Dv + dv];
            }

            state_local[x] = next_state;
        }
        __syncthreads();
    }

    // Store final state.
    CGDR_FOR(i, Dk * Dv) {
        int dk = i / Dv;
        int dv = i % Dv;

        const ptrdiff_t s2 = final_state_indices != nullptr ? initial_s2 : final_s2;
        const ptrdiff_t s3 = final_state_indices != nullptr ? initial_s3 : final_s3;
        ptrdiff_t write_idx = final_base + static_cast<ptrdiff_t>(dv) * s2 + static_cast<ptrdiff_t>(dk) * s3;

        final_state_target[write_idx] = static_cast<Tdata>(state_local[i]);
    }
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

    // One thread owns one value column for the whole sequence. Keeping its
    // recurrent state in registers avoids three global state passes per token.
    static_assert(NUM_THREADS == Dv, "register state requires one thread per value column");
    const int dv = threadIdx.x;
    Tcompute state_col[Dk];
    (void)state_workspace;
    __shared__ Tcompute q_vec[Dk];
    __shared__ Tcompute k_vec[Dk];
    __shared__ Tcompute scalar_buf[3];

    const int token_batch = has_cu_seqlens ? 0 : batch_idx;
    const Tcompute scale = rsqrtf(static_cast<Tcompute>(Dk));

    #pragma unroll
    for (int dk = 0; dk < static_cast<int>(Dk); ++dk) {
        const ptrdiff_t read_idx = initial_base + static_cast<ptrdiff_t>(dv) * initial_s2
                                  + static_cast<ptrdiff_t>(dk) * initial_s3;
        state_col[dk] = static_cast<Tcompute>(loadAsFloat(initial_state, read_idx));
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

        Tcompute projected = 0;
        #pragma unroll
        for (int dk = 0; dk < static_cast<int>(Dk); ++dk) {
            projected += k_vec[dk] * state_col[dk];
        }
        const ptrdiff_t v_base = static_cast<ptrdiff_t>(token_batch) * v_s0
                                + static_cast<ptrdiff_t>(token_idx) * v_s1
                                + static_cast<ptrdiff_t>(value_head_idx) * v_s2;
        const Tcompute delta = beta_t * (static_cast<Tcompute>(loadAsFloat(v, v_base + dv)) - decay * projected);
        Tcompute out_val = 0;
        #pragma unroll
        for (int dk = 0; dk < static_cast<int>(Dk); ++dk) {
            state_col[dk] = decay * state_col[dk] + k_vec[dk] * delta;
            out_val += q_vec[dk] * state_col[dk];
        }
        const ptrdiff_t out_base = static_cast<ptrdiff_t>(token_batch) * out_s0
                                  + static_cast<ptrdiff_t>(token_idx) * out_s1
                                  + static_cast<ptrdiff_t>(value_head_idx) * out_s2;
        out[out_base + dv] = static_cast<Tdata>(out_val);
        // All columns must finish reading shared Q/K before the next token.
        __syncthreads();
    }

    #pragma unroll
    for (int dk = 0; dk < static_cast<int>(Dk); ++dk) {
        const ptrdiff_t s2 = final_state_indices != nullptr ? initial_s2 : final_s2;
        const ptrdiff_t s3 = final_state_indices != nullptr ? initial_s3 : final_s3;
        const ptrdiff_t write_idx = final_base + static_cast<ptrdiff_t>(dv) * s2 + static_cast<ptrdiff_t>(dk) * s3;
        final_state_target[write_idx] = static_cast<Tdata>(state_col[dk]);
    }

}

#endif // __CHUNK_GATED_DELTA_RULE_KERNEL_CUH__
