// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM at commit 2f75e7f712fb2a013ce05ff357d94135231c8ae2:
// csrc/libtorch_stable/quantization/w8a8/cutlass/moe/moe_data.cu

#ifndef INFINI_OPS_NVIDIA_GET_CUTLASS_MOE_MM_DATA_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_GET_CUTLASS_MOE_MM_DATA_KERNEL_CUH_

#include <cstdint>

namespace infini::ops::get_cutlass_moe_mm_data_detail {

constexpr int32_t kBlockscaleAlignment = 128;
constexpr int64_t kSwapAbThreshold = 64;

__global__ void GetCutlassMoeMmDataKernel(
    const int32_t* __restrict__ topk_ids, int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ problem_sizes1, int32_t* __restrict__ problem_sizes2,
    int32_t* __restrict__ input_permutation,
    int32_t* __restrict__ output_permutation,
    int32_t* __restrict__ blockscale_offsets, int64_t numel, int32_t topk,
    int32_t num_experts, int32_t n, int32_t k, bool is_gated) {
  const bool swap_ab =
      blockscale_offsets == nullptr && numel <= kSwapAbThreshold;
  const int32_t count_index = swap_ab ? 1 : 0;
  const int32_t n1 = is_gated ? 2 * n : n;

  for (int32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    if (swap_ab) {
      problem_sizes1[expert * 3] = n1;
      problem_sizes1[expert * 3 + 1] = 0;
      problem_sizes1[expert * 3 + 2] = k;
      problem_sizes2[expert * 3] = k;
      problem_sizes2[expert * 3 + 1] = 0;
      problem_sizes2[expert * 3 + 2] = n;
    } else {
      problem_sizes1[expert * 3] = 0;
      problem_sizes1[expert * 3 + 1] = n1;
      problem_sizes1[expert * 3 + 2] = k;
      problem_sizes2[expert * 3] = 0;
      problem_sizes2[expert * 3 + 1] = k;
      problem_sizes2[expert * 3 + 2] = n;
    }
  }
  for (int64_t i = threadIdx.x; i < numel; i += blockDim.x) {
    input_permutation[i] = -1;
    output_permutation[i] = -1;
  }
  __syncthreads();

  for (int64_t i = threadIdx.x; i < numel; i += blockDim.x) {
    const int32_t expert = topk_ids[i];
    if (expert >= 0 && expert < num_experts) {
      atomicAdd(&problem_sizes1[expert * 3 + count_index], 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t offset = 0;
    int32_t blockscale_offset = 0;
    expert_offsets[0] = 0;
    if (blockscale_offsets != nullptr) {
      blockscale_offsets[0] = 0;
    }

    for (int32_t expert = 0; expert < num_experts; ++expert) {
      const int32_t count = problem_sizes1[expert * 3 + count_index];
      problem_sizes2[expert * 3 + count_index] = offset;
      offset += count;
      expert_offsets[expert + 1] = offset;

      if (blockscale_offsets != nullptr) {
        blockscale_offset += (count + kBlockscaleAlignment - 1) /
                             kBlockscaleAlignment * kBlockscaleAlignment;
        blockscale_offsets[expert + 1] = blockscale_offset;
      }
    }
  }
  __syncthreads();

  for (int64_t i = threadIdx.x; i < numel; i += blockDim.x) {
    const int32_t expert = topk_ids[i];
    if (expert == -1) {
      output_permutation[i] = expert_offsets[num_experts];
    } else if (expert >= 0 && expert < num_experts) {
      const int32_t destination =
          atomicAdd(&problem_sizes2[expert * 3 + count_index], 1);
      input_permutation[destination] = static_cast<int32_t>(i / topk);
      output_permutation[i] = destination;
    }
  }
  __syncthreads();

  for (int32_t expert = threadIdx.x; expert < num_experts;
       expert += blockDim.x) {
    problem_sizes2[expert * 3 + count_index] =
        problem_sizes1[expert * 3 + count_index];
  }
}

}  // namespace infini::ops::get_cutlass_moe_mm_data_detail

#endif  // INFINI_OPS_NVIDIA_GET_CUTLASS_MOE_MM_DATA_KERNEL_CUH_
