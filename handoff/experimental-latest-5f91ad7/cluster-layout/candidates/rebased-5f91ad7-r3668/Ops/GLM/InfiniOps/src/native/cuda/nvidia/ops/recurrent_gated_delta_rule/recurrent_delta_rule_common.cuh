// Ported from InfiniCore 1ab85ef1d3c2bce49edf1fc73a5b48becaf7552e:
// src/infiniop/ops/recurrent_gated_delta_rule/cuda/recurrent_delta_rule_common.cuh.
// Modified for InfiniOps: NVIDIA single-token decode and backend integration.
// Copyright 2025 InfiniTensor; MIT notice: LICENSES/InfiniCore-MIT.txt.
#ifndef __RECURRENT_DELTA_RULE_COMMON_CUH__
#define __RECURRENT_DELTA_RULE_COMMON_CUH__

#include <cmath>
#include <cstdint>

// Retain the original NVIDIA warp width.
#define INFINIOP_RECURRENT_DELTA_RULE_WARP_SIZE 32
#define INFINIOP_RECURRENT_DELTA_RULE_FULL_MASK 0xffffffffU

namespace op::recurrent_gated_delta_rule::cuda {

template <typename T>
__device__ inline float loadAsFloat(const T* ptr, ptrdiff_t offset) {
  return static_cast<float>(ptr[offset]);
}

template <>
__device__ inline float loadAsFloat<half>(const half* ptr, ptrdiff_t offset) {
  return __half2float(ptr[offset]);
}

template <>
__device__ inline float loadAsFloat<__nv_bfloat16>(const __nv_bfloat16* ptr,
                                                   ptrdiff_t offset) {
  return __bfloat162float(ptr[offset]);
}

__device__ inline int64_t loadOptionalIndex(const void* indices, bool is_i64,
                                            int index, int fallback) {
  if (indices == nullptr) {
    return static_cast<int64_t>(fallback);
  }
  return is_i64 ? static_cast<const int64_t*>(indices)[index]
                : static_cast<int64_t>(
                      static_cast<const int32_t*>(indices)[index]);
}

template <typename Tcompute>
__device__ inline Tcompute warpReduceSum(Tcompute value) {
  constexpr int WARP_SIZE = INFINIOP_RECURRENT_DELTA_RULE_WARP_SIZE;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(INFINIOP_RECURRENT_DELTA_RULE_FULL_MASK, value,
                              offset, WARP_SIZE);
  }
  return __shfl_sync(INFINIOP_RECURRENT_DELTA_RULE_FULL_MASK, value, 0,
                     WARP_SIZE);
}

template <typename Tcompute>
__device__ inline Tcompute blockReduceSum(Tcompute value, Tcompute* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const Tcompute result = scratch[0];
  __syncthreads();
  return result;
}

// The NVIDIA provider submits one decode token per batch row.
template <typename Tdata, typename Tcompute, size_t Dk, size_t Dv,
          size_t WARPS_PER_BLOCK, typename GatePolicy>
__device__ void recurrentDeltaRuleWarpSequence(
    Tdata* out, Tdata* initial_state, Tdata* final_state, const Tdata* q,
    const Tdata* k, const Tdata* v, const void* initial_state_indices,
    const void* final_state_indices, bool initial_state_indices_i64,
    bool final_state_indices_i64, bool use_qk_l2norm, bool indexed_state_pool,
    size_t pool_size, size_t num_key_heads, size_t value_heads_per_key_head,
    Tcompute query_scale, ptrdiff_t out_s0, ptrdiff_t out_s1, ptrdiff_t out_s2,
    ptrdiff_t initial_s0, ptrdiff_t initial_s1, ptrdiff_t initial_s2,
    ptrdiff_t initial_s3, ptrdiff_t final_s0, ptrdiff_t final_s1,
    ptrdiff_t final_s2, ptrdiff_t final_s3, ptrdiff_t q_s0, ptrdiff_t q_s1,
    ptrdiff_t q_s2, ptrdiff_t k_s0, ptrdiff_t k_s1, ptrdiff_t k_s2,
    ptrdiff_t v_s0, ptrdiff_t v_s1, ptrdiff_t v_s2, GatePolicy gate_policy,
    Tcompute* shared) {
  constexpr int WARP_SIZE = INFINIOP_RECURRENT_DELTA_RULE_WARP_SIZE;
  constexpr int NUM_THREADS = WARPS_PER_BLOCK * WARP_SIZE;
  constexpr int STATE_VALUES_PER_LANE = (Dk + WARP_SIZE - 1) / WARP_SIZE;

  const int batch_idx = blockIdx.x;
  const int value_head_idx = blockIdx.y;
  const int key_head_idx =
      value_head_idx / static_cast<int>(value_heads_per_key_head);
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane_idx = threadIdx.x & (WARP_SIZE - 1);
  const int value_dim_idx = blockIdx.z * WARPS_PER_BLOCK + warp_idx;
  const bool valid_value_dim = value_dim_idx < static_cast<int>(Dv);

  if (key_head_idx >= static_cast<int>(num_key_heads)) {
    return;
  }

  int64_t read_slot = batch_idx;
  int64_t write_slot = batch_idx;
  if (indexed_state_pool) {
    read_slot = loadOptionalIndex(
        initial_state_indices, initial_state_indices_i64, batch_idx, batch_idx);
    write_slot =
        final_state_indices == nullptr
            ? static_cast<int64_t>(batch_idx)
            : loadOptionalIndex(final_state_indices, final_state_indices_i64,
                                batch_idx, batch_idx);
    if (read_slot < 0 || read_slot >= static_cast<int64_t>(pool_size) ||
        (final_state_indices != nullptr &&
         (write_slot < 0 || write_slot >= static_cast<int64_t>(pool_size)))) {
      if (valid_value_dim && lane_idx == 0) {
        const ptrdiff_t out_base =
            static_cast<ptrdiff_t>(batch_idx) * out_s0 +
            static_cast<ptrdiff_t>(value_head_idx) * out_s2;
        out[out_base + value_dim_idx] = static_cast<Tdata>(0.0f);
      }
      return;
    }
  }

  const ptrdiff_t initial_base =
      static_cast<ptrdiff_t>(read_slot) * initial_s0 +
      static_cast<ptrdiff_t>(value_head_idx) * initial_s1 +
      static_cast<ptrdiff_t>(value_dim_idx) * initial_s2;

  Tdata* final_state_target =
      final_state_indices == nullptr ? final_state : initial_state;
  const ptrdiff_t final_base =
      final_state_indices == nullptr
          ? static_cast<ptrdiff_t>(batch_idx) * final_s0 +
                static_cast<ptrdiff_t>(value_head_idx) * final_s1 +
                static_cast<ptrdiff_t>(value_dim_idx) * final_s2
          : static_cast<ptrdiff_t>(write_slot) * initial_s0 +
                static_cast<ptrdiff_t>(value_head_idx) * initial_s1 +
                static_cast<ptrdiff_t>(value_dim_idx) * initial_s2;
  const ptrdiff_t final_k_stride =
      final_state_indices == nullptr ? final_s3 : initial_s3;

  Tcompute state[STATE_VALUES_PER_LANE];
#pragma unroll
  for (int i = 0; i < STATE_VALUES_PER_LANE; ++i) {
    const int key_dim_idx = lane_idx + i * WARP_SIZE;
    state[i] = valid_value_dim && key_dim_idx < static_cast<int>(Dk)
                   ? static_cast<Tcompute>(loadAsFloat(
                         initial_state,
                         initial_base +
                             static_cast<ptrdiff_t>(key_dim_idx) * initial_s3))
                   : static_cast<Tcompute>(0);
  }

  Tcompute* q_local = shared;
  Tcompute* k_local = q_local + Dk;
  Tcompute* decay_local = k_local + Dk;
  Tcompute* reduction_scratch = decay_local + Dk;
  Tcompute* beta_shared = reduction_scratch + NUM_THREADS;

  const int token_batch = batch_idx;
  {
    constexpr int64_t token_idx = 0;
    const ptrdiff_t q_base = static_cast<ptrdiff_t>(token_batch) * q_s0 +
                             static_cast<ptrdiff_t>(token_idx) * q_s1 +
                             static_cast<ptrdiff_t>(key_head_idx) * q_s2;
    const ptrdiff_t k_base = static_cast<ptrdiff_t>(token_batch) * k_s0 +
                             static_cast<ptrdiff_t>(token_idx) * k_s1 +
                             static_cast<ptrdiff_t>(key_head_idx) * k_s2;

    Tcompute q_sum = 0;
    Tcompute k_sum = 0;
    for (int key_dim_idx = threadIdx.x; key_dim_idx < static_cast<int>(Dk);
         key_dim_idx += NUM_THREADS) {
      const Tcompute q_value =
          static_cast<Tcompute>(loadAsFloat(q, q_base + key_dim_idx));
      const Tcompute k_value =
          static_cast<Tcompute>(loadAsFloat(k, k_base + key_dim_idx));
      q_local[key_dim_idx] = q_value;
      k_local[key_dim_idx] = k_value;
      q_sum += q_value * q_value;
      k_sum += k_value * k_value;
    }
    q_sum = blockReduceSum(q_sum, reduction_scratch);
    k_sum = blockReduceSum(k_sum, reduction_scratch);

    const Tcompute q_norm = use_qk_l2norm
                                ? rsqrtf(q_sum + static_cast<Tcompute>(1e-6))
                                : static_cast<Tcompute>(1);
    const Tcompute k_norm = use_qk_l2norm
                                ? rsqrtf(k_sum + static_cast<Tcompute>(1e-6))
                                : static_cast<Tcompute>(1);
    for (int key_dim_idx = threadIdx.x; key_dim_idx < static_cast<int>(Dk);
         key_dim_idx += NUM_THREADS) {
      q_local[key_dim_idx] *= q_norm * query_scale;
      k_local[key_dim_idx] *= k_norm;
    }

    gate_policy.prepare(token_batch, token_idx, key_head_idx, value_head_idx,
                        decay_local, beta_shared);
    __syncthreads();

    Tcompute kv_memory = 0;
    Tcompute hq_memory = 0;
    Tcompute kq_memory = 0;
#pragma unroll
    for (int i = 0; i < STATE_VALUES_PER_LANE; ++i) {
      const int key_dim_idx = lane_idx + i * WARP_SIZE;
      if (valid_value_dim && key_dim_idx < static_cast<int>(Dk)) {
        const Tcompute decayed_state = state[i] * decay_local[key_dim_idx];
        const Tcompute k_value = k_local[key_dim_idx];
        const Tcompute q_value = q_local[key_dim_idx];
        kv_memory += decayed_state * k_value;
        hq_memory += decayed_state * q_value;
        kq_memory += k_value * q_value;
      }
    }
    kv_memory = warpReduceSum(kv_memory);
    hq_memory = warpReduceSum(hq_memory);
    kq_memory = warpReduceSum(kq_memory);

    Tcompute delta = 0;
    if (valid_value_dim && lane_idx == 0) {
      const ptrdiff_t v_base = static_cast<ptrdiff_t>(token_batch) * v_s0 +
                               static_cast<ptrdiff_t>(token_idx) * v_s1 +
                               static_cast<ptrdiff_t>(value_head_idx) * v_s2;
      const Tcompute v_value =
          static_cast<Tcompute>(loadAsFloat(v, v_base + value_dim_idx));
      delta = (v_value - kv_memory) * beta_shared[0];
      const ptrdiff_t out_base =
          static_cast<ptrdiff_t>(token_batch) * out_s0 +
          static_cast<ptrdiff_t>(token_idx) * out_s1 +
          static_cast<ptrdiff_t>(value_head_idx) * out_s2;
      out[out_base + value_dim_idx] =
          static_cast<Tdata>(hq_memory + delta * kq_memory);
    }
    delta = __shfl_sync(INFINIOP_RECURRENT_DELTA_RULE_FULL_MASK, delta, 0,
                        WARP_SIZE);

#pragma unroll
    for (int i = 0; i < STATE_VALUES_PER_LANE; ++i) {
      const int key_dim_idx = lane_idx + i * WARP_SIZE;
      if (valid_value_dim && key_dim_idx < static_cast<int>(Dk)) {
        state[i] =
            state[i] * decay_local[key_dim_idx] + k_local[key_dim_idx] * delta;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < STATE_VALUES_PER_LANE; ++i) {
    const int key_dim_idx = lane_idx + i * WARP_SIZE;
    if (valid_value_dim && key_dim_idx < static_cast<int>(Dk)) {
      final_state_target[final_base +
                         static_cast<ptrdiff_t>(key_dim_idx) * final_k_stride] =
          static_cast<Tdata>(state[i]);
    }
  }
}

}  // namespace op::recurrent_gated_delta_rule::cuda

#endif  // __RECURRENT_DELTA_RULE_COMMON_CUH__
