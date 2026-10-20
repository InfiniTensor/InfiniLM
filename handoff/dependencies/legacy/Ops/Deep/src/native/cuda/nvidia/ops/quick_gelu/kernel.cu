#include "native/cuda/nvidia/ops/quick_gelu/kernel.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace infini::ops::nvidia_quick_gelu_detail {
namespace {
// The three branches preserve Core 0.2.9 quickgelu/cuda/kernel.cuh semantics.
// In particular FP16 must not be promoted to the BF16 FP32-intermediate path.
template <typename T>
__device__ __forceinline__ T Apply(T x) {
  constexpr float alpha = 1.702f;
  if constexpr (std::is_same_v<T, __half>) {
    const __half ax = __hmul(__float2half(alpha), x);
    const __half denominator = __hadd(__float2half(1.0f), hexp(__hneg(ax)));
    const __half sigmoid = hrcp(denominator);
    return __hmul(x, sigmoid);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    const float xf = __bfloat162float(x);
    const float ax = alpha * xf;
    const float sigmoid = 1.0f / (1.0f + __expf(-ax));
    return __float2bfloat16(xf * sigmoid);
  } else {
    const float ax = alpha * x;
    float sigmoid;
    if (ax >= 0.0f) {
      const float z = expf(-ax);
      sigmoid = 1.0f / (1.0f + z);
    } else {
      const float z = expf(ax);
      sigmoid = z / (1.0f + z);
    }
    return x * sigmoid;
  }
}

template <typename T>
__global__ void QuickGeluKernel(const T* input, T* out, Layout layout,
                                std::size_t count) {
  const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
           threadIdx.x; index < count; index += step) {
    std::size_t remainder = index, input_offset = 0, out_offset = 0;
    for (unsigned axis = layout.rank; axis > 0; --axis) {
      const auto coordinate = remainder % layout.shape[axis - 1];
      remainder /= layout.shape[axis - 1];
      input_offset += coordinate * layout.input_strides[axis - 1];
      out_offset += coordinate * layout.out_strides[axis - 1];
    }
    // No restrict qualifiers: exact non-overlapping in-place is supported.
    out[out_offset] = Apply(input[input_offset]);
  }
}

template <typename T>
void Submit(const void* input, void* out, Layout layout, std::size_t count,
            cudaStream_t stream) {
  constexpr unsigned threads = 256;
  const auto blocks = std::min<std::size_t>(1 + (count - 1) / threads, 65535);
  QuickGeluKernel<T><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
      static_cast<const T*>(input), static_cast<T*>(out), layout, count);
}
}  // namespace

void Launch(const void* input, void* out, DataType dtype, Layout layout,
            std::size_t count, cudaStream_t stream) {
  switch (dtype) {
    case DataType::kFloat16: Submit<__half>(input, out, layout, count, stream); break;
    case DataType::kBFloat16: Submit<__nv_bfloat16>(input, out, layout, count, stream); break;
    case DataType::kFloat32: Submit<float>(input, out, layout, count, stream); break;
    default: throw std::invalid_argument("NVIDIA QuickGelu unsupported dtype");
  }
  const auto status = cudaPeekAtLastError();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace infini::ops::nvidia_quick_gelu_detail
