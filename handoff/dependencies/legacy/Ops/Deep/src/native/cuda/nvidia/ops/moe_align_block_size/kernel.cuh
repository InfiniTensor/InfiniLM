#ifndef INFINI_OPS_NVIDIA_MOE_ALIGN_BLOCK_SIZE_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_MOE_ALIGN_BLOCK_SIZE_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

namespace infini::ops {
namespace moe_align_block_size_detail {

__device__ __forceinline__ int32_t MapExpertId(int32_t expert_id,
                                               const int32_t* expert_map,
                                               int32_t num_experts,
                                               int32_t map_size) {
  if (expert_id < 0 || expert_id >= (expert_map ? map_size : num_experts)) {
    return -1;
  }
  if (expert_map != nullptr) {
    expert_id = expert_map[expert_id];
  }

  return expert_id >= 0 && expert_id < num_experts ? expert_id : -1;
}

__global__ void MoeAlignBlockSizeKernel(
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_map,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ experts_ids,
    int32_t* __restrict__ num_tokens_post_pad, size_t numel,
    int32_t num_experts, int32_t block_size, size_t sorted_token_ids_size,
    size_t experts_ids_size, int32_t map_size) {
  extern __shared__ int32_t expert_offsets[];

  for (size_t i = threadIdx.x; i < sorted_token_ids_size; i += blockDim.x) {
    sorted_token_ids[i] = static_cast<int32_t>(numel);
  }
  for (size_t i = threadIdx.x; i < experts_ids_size; i += blockDim.x) {
    experts_ids[i] = -1;
  }
  for (int32_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
    expert_offsets[i] = 0;
  }
  if (threadIdx.x == 0) {
    *num_tokens_post_pad = 0;
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < numel; i += blockDim.x) {
    const auto expert_id = MapExpertId(topk_ids[i], expert_map,
                                       num_experts, map_size);
    if (expert_id >= 0) {
      atomicAdd(&expert_offsets[expert_id], 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t token_offset = 0;
    for (int32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
      const int32_t count = expert_offsets[expert_id];
      expert_offsets[expert_id] = token_offset;
      const int32_t padded_count =
          (count + block_size - 1) / block_size * block_size;
      for (int32_t i = 0; i < padded_count; i += block_size) {
        experts_ids[(token_offset + i) / block_size] = expert_id;
      }
      token_offset += padded_count;
    }
    *num_tokens_post_pad = token_offset;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (size_t i = 0; i < numel; ++i) {
      const auto expert_id = MapExpertId(topk_ids[i], expert_map,
                                         num_experts, map_size);
      if (expert_id >= 0) {
        sorted_token_ids[expert_offsets[expert_id]++] = static_cast<int32_t>(i);
      }
    }
  }
}

}  // namespace moe_align_block_size_detail
}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_MOE_ALIGN_BLOCK_SIZE_KERNEL_CUH_
