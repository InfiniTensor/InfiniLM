#ifndef INFINI_OPS_NVIDIA_GROUPED_TOPK_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_GROUPED_TOPK_KERNEL_CUH_

/*
 * Semantics and routing behavior are adapted from vLLM's `grouped_topk` CUDA
 * implementation:
 * https://github.com/vllm-project/vllm/blob/ffc4f08c8ee130d4ea6347c1bf31ffd4f8af28ab/csrc/libtorch_stable/moe/grouped_topk_kernels.cu
 * which is adapted from NVIDIA TensorRT-LLM's `noAuxTcKernels.cu`.
 * Copyright (c) 2025, The vLLM team.
 * Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <math_constants.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "native/cuda/nvidia/caster.cuh"

namespace infini::ops::grouped_topk_detail {

constexpr unsigned int kBlockSize = 256;
constexpr unsigned int kWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xffffffff;
constexpr int32_t kMaxNumGroups = 32;

template <typename T>
__device__ __forceinline__ float ToFloat(T value) {
  return Caster<Device::Type::kNvidia>::template Cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T FromFloat(float value) {
  return Caster<Device::Type::kNvidia>::template Cast<T>(value);
}

template <typename Score, int kScoringFunc>
__device__ __forceinline__ Score ApplyScoring(Score value) {
  if constexpr (kScoringFunc == 0) {
    return value;
  } else {
    const float input = ToFloat(value);
    return FromFloat<Score>(0.5f * tanhf(0.5f * input) + 0.5f);
  }
}

template <typename Score, typename Bias, int kScoringFunc>
__device__ __forceinline__ float SelectionScore(Score score, Bias bias) {
  const auto routed_score = ApplyScoring<Score, kScoringFunc>(score);
  const auto routed_bias = FromFloat<Score>(ToFloat(bias));
  return ToFloat(
      FromFloat<Score>(ToFloat(routed_score) + ToFloat(routed_bias)));
}

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

__device__ __forceinline__ void InsertTopTwo(float value, float& largest,
                                             float& second_largest) {
  if (value > largest) {
    second_largest = largest;
    largest = value;
  } else if (value > second_largest) {
    second_largest = value;
  }
}

template <unsigned int kSize>
__device__ __forceinline__ void BlockBest(float& value, int32_t& index) {
  __shared__ float values[kSize];
  __shared__ int32_t indices[kSize];
  values[threadIdx.x] = value;
  indices[threadIdx.x] = index;
  __syncthreads();

  for (unsigned int stride = kSize / 2; stride > 0; stride >>= 1) {
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

template <typename Score, typename Bias, int kScoringFunc>
__device__ __forceinline__ float GroupScore(const Score* scores,
                                            const Bias* bias, int32_t group,
                                            int32_t num_experts_per_group,
                                            int32_t lane) {
  const int32_t first_expert = group * num_experts_per_group;
  float largest = -CUDART_INF_F;
  float second_largest = -CUDART_INF_F;

  for (int32_t local_expert = lane; local_expert < num_experts_per_group;
       local_expert += kWarpSize) {
    const int32_t expert = first_expert + local_expert;
    InsertTopTwo(
        SelectionScore<Score, Bias, kScoringFunc>(scores[expert], bias[expert]),
        largest, second_largest);
  }

  for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const float other_largest =
        __shfl_down_sync(kFullWarpMask, largest, offset);
    const float other_second =
        __shfl_down_sync(kFullWarpMask, second_largest, offset);
    if (lane + offset < static_cast<int32_t>(kWarpSize)) {
      InsertTopTwo(other_largest, largest, second_largest);
      InsertTopTwo(other_second, largest, second_largest);
    }
  }

  return ToFloat(FromFloat<Score>(largest + second_largest));
}

template <typename Score, typename Bias, int kScoringFunc>
__global__ void GroupedTopkKernel(
    const Score* __restrict__ scores, const Bias* __restrict__ bias,
    float* __restrict__ topk_values, int32_t* __restrict__ topk_indices,
    int32_t num_experts, int32_t num_expert_group, int32_t topk_group,
    int32_t topk, bool renormalize, double routed_scaling_factor) {
  __shared__ float group_scores[kMaxNumGroups];
  __shared__ int32_t selected_groups[kMaxNumGroups];
  __shared__ int32_t proceed;

  const int32_t token = static_cast<int32_t>(blockIdx.x);
  const int32_t thread = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = thread / static_cast<int32_t>(kWarpSize);
  const int32_t lane = thread % static_cast<int32_t>(kWarpSize);
  const int32_t num_warps = static_cast<int32_t>(kBlockSize / kWarpSize);
  const int32_t num_experts_per_group = num_experts / num_expert_group;
  const int64_t scores_offset = static_cast<int64_t>(token) * num_experts;
  const int64_t output_offset = static_cast<int64_t>(token) * topk;
  const auto token_scores = scores + scores_offset;

  if (thread < kMaxNumGroups) {
    group_scores[thread] = -CUDART_INF_F;
    selected_groups[thread] = 0;
  }
  __syncthreads();

  for (int32_t group = warp; group < num_expert_group; group += num_warps) {
    const float score = GroupScore<Score, Bias, kScoringFunc>(
        token_scores, bias, group, num_experts_per_group, lane);
    if (lane == 0) {
      group_scores[group] = score;
    }
  }
  __syncthreads();

  if (thread == 0) {
    proceed = 1;
    for (int32_t rank = 0; rank < topk_group; ++rank) {
      float best_value = -CUDART_INF_F;
      int32_t best_group = -1;
      for (int32_t group = 0; group < num_expert_group; ++group) {
        if (!selected_groups[group] &&
            IsBetter(group_scores[group], group, best_value, best_group)) {
          best_value = group_scores[group];
          best_group = group;
        }
      }
      if (best_group < 0 || best_value == -CUDART_INF_F) {
        proceed = 0;
        break;
      }
      selected_groups[best_group] = 1;
    }
  }
  __syncthreads();

  if (!proceed) {
    if (thread < topk) {
      topk_indices[output_offset + thread] = thread;
      topk_values[output_offset + thread] = 1.0f / static_cast<float>(topk);
    }
    return;
  }

  for (int32_t rank = 0; rank < topk; ++rank) {
    float best_value = -CUDART_INF_F;
    int32_t best_expert = -1;

    for (int32_t expert = thread; expert < num_experts;
         expert += static_cast<int32_t>(kBlockSize)) {
      if (!selected_groups[expert / num_experts_per_group]) {
        continue;
      }

      bool already_selected = false;
      for (int32_t prior_rank = 0; prior_rank < rank; ++prior_rank) {
        if (topk_indices[output_offset + prior_rank] == expert) {
          already_selected = true;
          break;
        }
      }
      if (already_selected || !isfinite(ToFloat(token_scores[expert]))) {
        continue;
      }

      const float selection_score = SelectionScore<Score, Bias, kScoringFunc>(
          token_scores[expert], bias[expert]);
      if (IsBetter(selection_score, expert, best_value, best_expert)) {
        best_value = selection_score;
        best_expert = expert;
      }
    }

    BlockBest<kBlockSize>(best_value, best_expert);
    if (thread == 0) {
      if (best_expert < 0) {
        best_expert = 0;
      }
      topk_indices[output_offset + rank] = best_expert;
      topk_values[output_offset + rank] =
          ToFloat(ApplyScoring<Score, kScoringFunc>(token_scores[best_expert]));
    }
    __syncthreads();
  }

  if (thread == 0) {
    float scale = static_cast<float>(routed_scaling_factor);
    if (renormalize) {
      float topk_sum = 1e-20f;
      for (int32_t rank = 0; rank < topk; ++rank) {
        topk_sum += topk_values[output_offset + rank];
      }
      scale /= topk_sum;
    }
    for (int32_t rank = 0; rank < topk; ++rank) {
      topk_values[output_offset + rank] *= scale;
    }
  }
}

}  // namespace infini::ops::grouped_topk_detail

#endif  // INFINI_OPS_NVIDIA_GROUPED_TOPK_KERNEL_CUH_
