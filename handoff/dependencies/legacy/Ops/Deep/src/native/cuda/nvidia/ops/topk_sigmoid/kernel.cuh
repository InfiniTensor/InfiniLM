#ifndef INFINI_OPS_NVIDIA_TOPK_SIGMOID_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_TOPK_SIGMOID_KERNEL_CUH_

/*
 * Portions of this kernel are adapted from vLLM's `topk_sigmoid` CUDA
 * implementation:
 * https://github.com/vllm-project/vllm/blob/88402a41c4ab272ebbbd33f4a77fbbac0431cbb9/csrc/libtorch_stable/moe/topk_softmax_kernels.cu
 * Copyright 2024 The vLLM team.
 * Copyright 1993-2023 NVIDIA Corporation and affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "native/cuda/caster.cuh"

namespace infini::ops {
namespace topk_sigmoid_detail {

__device__ __forceinline__ bool IsBetter(float value, int32_t index,
                                         float best_value, int32_t best_index) {
  if (index < 0) {
    return false;
  }
  if (best_index < 0) {
    return true;
  }
  return value > best_value || (value == best_value && index < best_index);
}

template <unsigned int kBlockSize>
__device__ __forceinline__ void BlockBest(float& value, int32_t& index) {
  __shared__ float values[kBlockSize];
  __shared__ int32_t indices[kBlockSize];
  values[threadIdx.x] = value;
  indices[threadIdx.x] = index;
  __syncthreads();

  for (unsigned int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride &&
        IsBetter(values[threadIdx.x + stride], indices[threadIdx.x + stride],
                 values[threadIdx.x], indices[threadIdx.x])) {
      values[threadIdx.x] = values[threadIdx.x + stride];
      indices[threadIdx.x] = indices[threadIdx.x + stride];
    }
    __syncthreads();
  }

  value = values[0];
  index = indices[0];
}

template <Device::Type kDevice, typename Input>
__device__ __forceinline__ float ToFloat(Input value) {
  return Caster<kDevice>::template Cast<float>(value);
}

template <unsigned int kBlockSize, Device::Type kDevice, typename Input,
          typename Index>
__global__ void TopkSigmoidKernel(
    const Input* __restrict__ gating_output,
    const float* __restrict__ e_score_correction_bias,
    const uint8_t* __restrict__ is_padding, float* __restrict__ topk_weights,
    Index* __restrict__ topk_ids, int32_t* __restrict__ token_expert_indices,
    int32_t num_tokens, int32_t num_experts, int32_t topk, bool renormalize,
    float routed_scaling_factor) {
  const int32_t token = static_cast<int32_t>(blockIdx.x);
  const int64_t input_offset = static_cast<int64_t>(token) * num_experts;
  const int64_t output_offset = static_cast<int64_t>(token) * topk;

  float selected_sum = 0.0f;
  for (int32_t rank = 0; rank < topk; ++rank) {
    float thread_best = -FLT_MAX;
    int32_t thread_index = -1;

    for (int32_t expert = static_cast<int32_t>(threadIdx.x);
         expert < num_experts; expert += kBlockSize) {
      bool selected = false;
      for (int32_t prior_rank = 0; prior_rank < rank; ++prior_rank) {
        if (static_cast<int64_t>(topk_ids[output_offset + prior_rank]) ==
            expert) {
          selected = true;
          break;
        }
      }
      if (selected) {
        continue;
      }

      const float input =
          ToFloat<kDevice>(gating_output[input_offset + expert]);
      float score = 1.0f / (1.0f + expf(-input));
      if (!isfinite(score)) {
        score = 0.0f;
      }
      const float selection_score =
          score +
          (e_score_correction_bias ? e_score_correction_bias[expert] : 0.0f);
      if (IsBetter(selection_score, expert, thread_best, thread_index)) {
        thread_best = selection_score;
        thread_index = expert;
      }
    }

    BlockBest<kBlockSize>(thread_best, thread_index);
    if (threadIdx.x == 0) {
      const float input =
          ToFloat<kDevice>(gating_output[input_offset + thread_index]);
      float weight = 1.0f / (1.0f + expf(-input));
      if (!isfinite(weight)) {
        weight = 0.0f;
      }
      topk_weights[output_offset + rank] = weight;
      topk_ids[output_offset + rank] = static_cast<Index>(thread_index);
      token_expert_indices[output_offset + rank] = rank * num_tokens + token;
      selected_sum += weight;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float scale = routed_scaling_factor;
    if (renormalize) {
      scale /= selected_sum > 0.0f ? selected_sum : 1.0f;
    }
    const bool padding = is_padding && is_padding[token] != 0;
    for (int32_t rank = 0; rank < topk; ++rank) {
      topk_weights[output_offset + rank] *= scale;
      if (padding) {
        topk_ids[output_offset + rank] = static_cast<Index>(-1);
      }
    }
  }
}

}  // namespace topk_sigmoid_detail
}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_TOPK_SIGMOID_KERNEL_CUH_
