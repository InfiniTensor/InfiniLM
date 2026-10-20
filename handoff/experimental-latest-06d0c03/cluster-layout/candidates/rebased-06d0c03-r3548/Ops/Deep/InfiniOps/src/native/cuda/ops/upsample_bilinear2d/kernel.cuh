#ifndef INFINI_OPS_CUDA_UPSAMPLE_BILINEAR2D_KERNEL_CUH_
#define INFINI_OPS_CUDA_UPSAMPLE_BILINEAR2D_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>

#include "native/cuda/nvidia/ops/upsample_bilinear2d/kernel.h"

namespace infini::ops::nvidia_upsample_bilinear2d_detail {
template <typename T>
__device__ float ToFloat(T value);
template <>
__device__ inline float ToFloat<float>(float value) {
  return value;
}
template <>
__device__ inline float ToFloat<__half>(__half value) {
  return __half2float(value);
}
template <>
__device__ inline float ToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ T FromFloat(float value);
template <>
__device__ inline float FromFloat<float>(float value) {
  return value;
}
template <>
__device__ inline __half FromFloat<__half>(float value) {
  return __float2half_rn(value);
}
template <>
__device__ inline __nv_bfloat16 FromFloat<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

__device__ inline std::size_t Clamp(std::int64_t value, std::size_t maximum) {
  if (value < 0) return 0;
  const auto positive = static_cast<std::size_t>(value);
  return positive > maximum ? maximum : positive;
}

template <typename T>
__global__ void UpsampleBilinear2dKernel(const T* input, T* out, Layout layout,
                                         std::size_t output_count) {
  const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto pos =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       pos < output_count; pos += step) {
    auto remainder = pos;
    const auto w = remainder % layout.output_width;
    remainder /= layout.output_width;
    const auto h = remainder % layout.output_height;
    remainder /= layout.output_height;
    const auto c = remainder % layout.channels;
    const auto n = remainder / layout.channels;

    const float source_h =
        layout.align_corners
            ? static_cast<float>(h) * layout.scale_height
            : (static_cast<float>(h) + 0.5F) * layout.scale_height - 0.5F;
    const float source_w =
        layout.align_corners
            ? static_cast<float>(w) * layout.scale_width
            : (static_cast<float>(w) + 0.5F) * layout.scale_width - 0.5F;
    const auto h0_raw = static_cast<std::int64_t>(floorf(source_h));
    const auto w0_raw = static_cast<std::int64_t>(floorf(source_w));
    const float h1_weight = source_h - static_cast<float>(h0_raw);
    const float w1_weight = source_w - static_cast<float>(w0_raw);
    const auto h0 = Clamp(h0_raw, layout.input_height - 1);
    const auto h1 = Clamp(h0_raw + 1, layout.input_height - 1);
    const auto w0 = Clamp(w0_raw, layout.input_width - 1);
    const auto w1 = Clamp(w0_raw + 1, layout.input_width - 1);
    const auto base = n * layout.input_strides[0] + c * layout.input_strides[1];
    const float v00 = ToFloat(input[base + h0 * layout.input_strides[2] +
                                    w0 * layout.input_strides[3]]);
    const float v01 = ToFloat(input[base + h0 * layout.input_strides[2] +
                                    w1 * layout.input_strides[3]]);
    const float v10 = ToFloat(input[base + h1 * layout.input_strides[2] +
                                    w0 * layout.input_strides[3]]);
    const float v11 = ToFloat(input[base + h1 * layout.input_strides[2] +
                                    w1 * layout.input_strides[3]]);
    const float top = (1.0F - w1_weight) * v00 + w1_weight * v01;
    const float bottom = (1.0F - w1_weight) * v10 + w1_weight * v11;
    out[pos] = FromFloat<T>((1.0F - h1_weight) * top + h1_weight * bottom);
  }
}
}  // namespace infini::ops::nvidia_upsample_bilinear2d_detail

#endif  // INFINI_OPS_CUDA_UPSAMPLE_BILINEAR2D_KERNEL_CUH_
