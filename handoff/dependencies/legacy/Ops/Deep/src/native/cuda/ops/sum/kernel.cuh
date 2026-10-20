#ifndef INFINI_OPS_CUDA_SUM_KERNEL_CUH_
#define INFINI_OPS_CUDA_SUM_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "native/cuda/nvidia/ops/sum/kernel.h"

namespace infini::ops::nvidia_sum_detail {
template <typename T>
__device__ float ToFloat(T value);
template <>
__device__ inline float ToFloat<float>(float value) { return value; }
template <>
__device__ inline float ToFloat<__half>(__half value) { return __half2float(value); }
template <>
__device__ inline float ToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ T FromFloat(float value);
template <>
__device__ inline float FromFloat<float>(float value) { return value; }
template <>
__device__ inline __half FromFloat<__half>(float value) {
  return __float2half_rn(value);
}
template <>
__device__ inline __nv_bfloat16 FromFloat<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void SumKernel(const T* input, T* out, Layout layout,
                          std::size_t output_count) {
  const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto pos = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       pos < output_count; pos += step) {
    auto remaining = pos;
    std::int64_t source = 0;
    for (int d = layout.rank - 1; d >= 0; --d) {
      if (d == layout.axis) continue;
      source += static_cast<std::int64_t>(remaining % layout.shape[d]) *
                layout.stride[d];
      remaining /= layout.shape[d];
    }
    float total = 0.0F;
    for (std::int64_t r = 0; r < layout.reduce_length; ++r) {
      total += ToFloat(input[source + r * layout.stride[layout.axis]]);
    }
    out[pos] = FromFloat<T>(total);
  }
}
}  // namespace infini::ops::nvidia_sum_detail

#endif  // INFINI_OPS_CUDA_SUM_KERNEL_CUH_
