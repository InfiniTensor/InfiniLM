#ifndef INFINI_OPS_CUDA_RESHAPE_AND_CACHE_FLASH_KERNEL_CUH_
#define INFINI_OPS_CUDA_RESHAPE_AND_CACHE_FLASH_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

namespace infini::ops {

template <typename Tdata, int kBlockSize>
__global__ void ReshapeAndCacheFlashKernel(
    Tdata* __restrict__ k_cache, Tdata* __restrict__ v_cache,
    const Tdata* __restrict__ k, const Tdata* __restrict__ v,
    const int64_t* __restrict__ slot_mapping, std::size_t head_size,
    std::size_t cache_block_size, std::ptrdiff_t k_src_stride,
    std::ptrdiff_t v_src_stride, std::ptrdiff_t k_src_head_stride,
    std::ptrdiff_t v_src_head_stride, std::ptrdiff_t k_cache_block_stride,
    std::ptrdiff_t v_cache_block_stride, std::ptrdiff_t k_cache_head_stride,
    std::ptrdiff_t v_cache_head_stride, std::ptrdiff_t k_cache_slot_stride,
    std::ptrdiff_t v_cache_slot_stride) {
  auto head_idx = static_cast<std::size_t>(blockIdx.x);
  auto token_idx = static_cast<std::size_t>(blockIdx.y);
  int64_t slot = slot_mapping[token_idx];

  if (slot < 0) {
    return;
  }

  auto physical_block_idx = static_cast<std::size_t>(slot) / cache_block_size;
  auto block_offset = static_cast<std::size_t>(slot) % cache_block_size;

  const Tdata* k_src =
      k + token_idx * k_src_stride + head_idx * k_src_head_stride;
  const Tdata* v_src =
      v + token_idx * v_src_stride + head_idx * v_src_head_stride;
  Tdata* k_dst = k_cache + physical_block_idx * k_cache_block_stride +
                 head_idx * k_cache_head_stride +
                 block_offset * k_cache_slot_stride;
  Tdata* v_dst = v_cache + physical_block_idx * v_cache_block_stride +
                 head_idx * v_cache_head_stride +
                 block_offset * v_cache_slot_stride;

  for (std::size_t i = threadIdx.x; i < head_size; i += kBlockSize) {
    k_dst[i] = k_src[i];
    v_dst[i] = v_src[i];
  }
}

}  // namespace infini::ops

#endif
