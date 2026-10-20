#include "native/cuda/nvidia/ops/mul_scalar/kernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

namespace infini::ops::nvidia_mul_scalar_detail {
namespace {
template <typename T>
__device__ __forceinline__ T Multiply(T value, T alpha) {
  return value * alpha;
}

template <>
__device__ __forceinline__ __half Multiply(__half value, __half alpha) {
  return __hmul(value, alpha);
}

template <>
__device__ __forceinline__ __nv_bfloat16 Multiply(
    __nv_bfloat16 value, __nv_bfloat16 alpha) {
  return __hmul(value, alpha);
}

template <typename T>
__global__ void MulScalarContiguousKernel(const T* input, T* out, T alpha,
                                          std::size_t count) {
  const auto start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t linear = start; linear < count; linear += step) {
    out[linear] = Multiply(input[linear], alpha);
  }
}

template <typename T>
__global__ void MulScalarStridedKernel(const T* input, T* out, Layout layout,
                                       T alpha, std::size_t count) {
  const auto start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t linear = start; linear < count; linear += step) {
    std::size_t remaining = linear;
    std::int64_t input_offset = 0, out_offset = 0;
    for (std::int64_t d = layout.ndim; d-- > 0;) {
      const auto coord = static_cast<std::int64_t>(
          remaining % static_cast<std::size_t>(layout.shape[d]));
      remaining /= static_cast<std::size_t>(layout.shape[d]);
      input_offset += coord * layout.input_strides[d];
      out_offset += coord * layout.out_strides[d];
    }
    out[out_offset] = Multiply(input[input_offset], alpha);
  }
}

template <typename T>
void Submit(const void* input, void* out, const Layout* layout, T alpha,
            std::size_t count, cudaStream_t stream) {
  constexpr unsigned kThreads = 256;
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count - 1) / kThreads + 1, 65535));
  if (!layout)
    MulScalarContiguousKernel<T><<<blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(out), alpha,
        count);
  else
    MulScalarStridedKernel<T><<<blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(out), *layout, alpha,
        count);
}
}  // namespace

void Launch(const void* input, void* out, DataType dtype, double alpha,
            const Layout* layout, std::size_t count, cudaStream_t stream) {
  if (!count) return;
  switch (dtype) {
    case DataType::kFloat16:
      Submit<__half>(input, out, layout,
                     __float2half_rn(static_cast<float>(alpha)), count, stream);
      break;
    case DataType::kBFloat16:
      Submit<__nv_bfloat16>(
          input, out, layout,
          __float2bfloat16_rn(static_cast<float>(alpha)), count, stream);
      break;
    case DataType::kFloat32:
      Submit<float>(input, out, layout, static_cast<float>(alpha), count,
                    stream);
      break;
    case DataType::kFloat64:
      Submit<double>(input, out, layout, alpha, count, stream);
      break;
    default:
      throw std::invalid_argument("NVIDIA MulScalar unsupported dtype");
  }
}
}  // namespace infini::ops::nvidia_mul_scalar_detail
