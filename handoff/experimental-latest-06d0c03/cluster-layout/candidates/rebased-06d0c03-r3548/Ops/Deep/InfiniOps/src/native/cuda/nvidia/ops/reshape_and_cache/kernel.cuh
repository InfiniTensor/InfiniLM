#ifndef INFINI_OPS_NVIDIA_RESHAPE_AND_CACHE_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_RESHAPE_AND_CACHE_KERNEL_CUH_

#include <cuda_fp8.h>

#include <cstddef>
#include <cstdint>

#include "native/cuda/caster.cuh"
#include "native/cuda/nvidia/caster.cuh"

namespace infini::ops {

template <typename T>
__device__ __forceinline__ uint8_t ToFp8Cache(T value, float scale,
                                              bool use_e5m2) {
  float converted =
      Caster<Device::Type::kNvidia>::template Cast<float>(value) / scale;
  auto format = use_e5m2 ? __NV_E5M2 : __NV_E4M3;
  return static_cast<uint8_t>(
      __nv_cvt_float_to_fp8(converted, __NV_SATFINITE, format));
}

template <typename T, typename TCache, bool kQuantized, bool kUseE5M2>
__global__ void ReshapeAndCacheKernel(
    const T* __restrict__ key, const T* __restrict__ value,
    TCache* __restrict__ key_cache, TCache* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, int64_t key_token_stride,
    int64_t value_token_stride, int64_t key_head_stride,
    int64_t value_head_stride, size_t num_heads, size_t head_size,
    size_t block_size, size_t x, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale) {
  const size_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const size_t block_idx = static_cast<size_t>(slot_idx) / block_size;
  const size_t block_offset = static_cast<size_t>(slot_idx) % block_size;
  const size_t head_blocks = head_size / x;

  for (size_t i = threadIdx.x; i < num_heads * head_size; i += blockDim.x) {
    const size_t head_idx = i / head_size;
    const size_t head_offset = i % head_size;
    const size_t head_block = head_offset / x;
    const size_t x_offset = head_offset % x;
    const size_t key_src_idx =
        token_idx * key_token_stride + head_idx * key_head_stride + head_offset;
    const size_t value_src_idx = token_idx * value_token_stride +
                                 head_idx * value_head_stride + head_offset;
    const size_t key_dst_idx =
        (((block_idx * num_heads + head_idx) * head_blocks + head_block) *
             block_size +
         block_offset) *
            x +
        x_offset;
    const size_t value_dst_idx =
        ((block_idx * num_heads + head_idx) * head_size + head_offset) *
            block_size +
        block_offset;

    if constexpr (kQuantized) {
      key_cache[key_dst_idx] = ToFp8Cache(key[key_src_idx], *k_scale, kUseE5M2);
      value_cache[value_dst_idx] =
          ToFp8Cache(value[value_src_idx], *v_scale, kUseE5M2);
    } else {
      key_cache[key_dst_idx] = key[key_src_idx];
      value_cache[value_dst_idx] = value[value_src_idx];
    }
  }
}

}  // namespace infini::ops

#endif
