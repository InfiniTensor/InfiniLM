/*
 * Portions of the CUDA kernels in this file are adapted from SGLang:
 * /sgl-kernel/csrc/moe/moe_fused_gate.cu
 *
 * Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include "native/cuda/hygon/ops/moe_fused_gate/kernel.h"
#include "native/cuda/hygon/ops/moe_fused_gate/device_guard.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
namespace infini::ops { namespace {

#define MOE_CHECK_CUDA(expr) do { auto s=(expr); if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); } while(0)

constexpr int MAX_VPT = 32;

template <typename T>
__device__ float convert_to_float(T x) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(x);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(x);
    } else if constexpr (std::is_same_v<T, float>) {
        return x;
    } else {
        return static_cast<float>(x);
    }
}

struct KernelParamsDynamic {
    int VPT;
    int NUM_EXPERTS;
    int THREADS_PER_ROW;
    int ROWS_PER_WARP;
    int ROWS_PER_CTA;
};

template <typename T, typename Params>
__device__ void moe_fused_gate_impl(
    const T *__restrict__ input,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int32_t *__restrict__ indices,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output,
    Params params) {
    const int tidx = threadIdx.x;
    const int64_t thread_row = blockIdx.x * params.ROWS_PER_CTA + threadIdx.y * params.ROWS_PER_WARP + tidx / params.THREADS_PER_ROW;
    if (thread_row >= num_rows) {
        return;
    }

    const int64_t routed_topk = topk - num_fused_shared_experts;
    const T *thread_row_ptr = input + thread_row * params.NUM_EXPERTS;
    const int thread_group_idx = tidx % params.THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * params.VPT;

    float row_chunk[MAX_VPT];
    float bias_chunk[MAX_VPT];
#pragma unroll
    for (int ii = 0; ii < MAX_VPT; ++ii) {
        if (ii < params.VPT) {
            const int expert = first_elt_read_by_thread + ii;
            const float sigmoid = 1.0f / (1.0f + expf(-convert_to_float(thread_row_ptr[expert])));
            row_chunk[ii] = sigmoid;
            bias_chunk[ii] = sigmoid + bias[expert];
        }
    }

    __syncthreads();

    for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group; ++k_idx) {
        int expert = first_elt_read_by_thread;
        float max_val = -FLT_MAX;
        float max_val_second = -FLT_MAX;
#pragma unroll
        for (int ii = 0; ii < MAX_VPT; ++ii) {
            if (ii < params.VPT) {
                const float val = bias_chunk[ii];
                if (val > max_val) {
                    max_val_second = max_val;
                    max_val = val;
                } else if (val > max_val_second) {
                    max_val_second = val;
                }
            }
        }
        float max_sum = max_val + max_val_second;

#pragma unroll
        for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            const float other_max_sum = __shfl_xor_sync(__activemask(), max_sum, mask, params.THREADS_PER_ROW);
            const int other_expert = __shfl_xor_sync(__activemask(), expert, mask, params.THREADS_PER_ROW);
            if (max_sum > other_max_sum || (other_max_sum == max_sum && other_expert > expert)) {
                max_sum = other_max_sum;
                expert = other_expert;
            }
        }

        const int thread_to_clear_in_group = expert / params.VPT;
        if (thread_group_idx == thread_to_clear_in_group) {
#pragma unroll
            for (int ii = 0; ii < MAX_VPT; ++ii) {
                if (ii < params.VPT) {
                    bias_chunk[ii] = FLT_MAX;
                }
            }
        }
    }

    __syncthreads();

    float output_sum = 0.0f;
    for (int k_idx = 0; k_idx < routed_topk; ++k_idx) {
        float max_val = bias_chunk[0];
        int expert = first_elt_read_by_thread;
        if (max_val != FLT_MAX) {
#pragma unroll
            for (int ii = 1; ii < MAX_VPT; ++ii) {
                if (ii < params.VPT) {
                    const float val = bias_chunk[ii];
                    if (val > max_val) {
                        max_val = val;
                        expert = first_elt_read_by_thread + ii;
                    }
                }
            }
        } else {
            max_val = -FLT_MAX;
        }

#pragma unroll
        for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            const float other_max = __shfl_xor_sync(__activemask(), max_val, mask, params.THREADS_PER_ROW);
            const int other_expert = __shfl_xor_sync(__activemask(), expert, mask, params.THREADS_PER_ROW);
            if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert = other_expert;
            }
        }

        const int thread_to_clear_in_group = expert / params.VPT;
        const int64_t idx = topk * thread_row + k_idx;
        if (thread_group_idx == thread_to_clear_in_group) {
            const int expert_to_clear_in_thread = expert % params.VPT;
            bias_chunk[expert_to_clear_in_thread] = -FLT_MAX;
            output[idx] = row_chunk[expert_to_clear_in_thread];
            indices[idx] = static_cast<int32_t>(expert);
        }

        __syncthreads();
        if (thread_group_idx == 0) {
            output_sum += output[idx];
        }
        __syncthreads();
    }

    if (thread_group_idx == 0 && num_fused_shared_experts > 0) {
        int64_t idx = topk * thread_row + routed_topk;
        for (int i = 0; i < num_fused_shared_experts; ++i) {
            indices[idx + i] = static_cast<int32_t>(params.NUM_EXPERTS + i);
            output[idx + i] = output_sum / routed_scaling_factor;
        }
    }
    __syncthreads();

    if (thread_group_idx == 0) {
#pragma unroll
        for (int ii = 0; ii < topk; ++ii) {
            const int64_t idx = topk * thread_row + ii;
            output[idx] = output[idx] / output_sum;
            if (apply_routed_scaling_factor_on_output) {
                output[idx] *= routed_scaling_factor;
            }
        }
    }
}

template <typename T>
__global__ void moe_fused_gate_kernel_dynamic(
    const T *__restrict__ input,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int32_t *__restrict__ indices,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
    KernelParamsDynamic params;
    params.NUM_EXPERTS = static_cast<int>(num_experts);
    params.VPT = static_cast<int>(num_experts / num_expert_group);
    params.THREADS_PER_ROW = static_cast<int>(num_expert_group);
    params.ROWS_PER_WARP = 1;
    params.ROWS_PER_CTA = 1;
    moe_fused_gate_impl<T>(
        input, bias, output, indices, num_rows, topk_group, topk, num_fused_shared_experts,
        routed_scaling_factor, apply_routed_scaling_factor_on_output, params);
}


} // namespace
void Operator<MoeFusedGate, Device::Type::kHygon, 0>::operator()(Tensor weights, Tensor indices, const Tensor input, const Tensor bias, size_t groups, size_t selected_groups, size_t shared, float scale, bool apply_scale) const {
  moe_detail::Require(input.device().type()==Device::Type::kHygon);
  MoeDeviceGuard device_guard{metadata_[0].device().index()};
  Invocation(weights, indices, input, bias, groups, selected_groups, shared, scale, apply_scale); if(input.size(0)==0) return;
  moe_detail::Require(!(input.size(1)&(input.size(1)-1)) && groups<=32 && !(groups&(groups-1)) && input.size(1)/groups<=32);
  const auto stream=static_cast<cudaStream_t>(stream_);
  auto run=[&](auto value) { using T=decltype(value);
    moe_fused_gate_kernel_dynamic<T><<<input.size(0), groups, 0, stream>>>(
      static_cast<const T*>(input.data()),static_cast<const float*>(bias.data()),
      static_cast<float*>(weights.data()),static_cast<int32_t*>(indices.data()),
      input.size(0),input.size(1),groups,selected_groups,weights.size(1),shared,scale,apply_scale);
  };
  switch(input.dtype()) { case DataType::kFloat16: run(half{}); break;
    case DataType::kBFloat16: run(__nv_bfloat16{}); break; case DataType::kFloat32: run(float{}); break;
    default: throw std::invalid_argument("Gate dtype"); }
  MOE_CHECK_CUDA(cudaGetLastError());
}

#undef MOE_CHECK_CUDA
} // namespace infini::ops
