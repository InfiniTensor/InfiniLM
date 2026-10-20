#ifndef INFINI_OPS_CUDA_RELU_KERNEL_CUH_
#define INFINI_OPS_CUDA_RELU_KERNEL_CUH_

#include <cstddef>
#include <type_traits>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

namespace {

template <Device::Type kDev, typename T>
__device__ __forceinline__ T ReluValue(T value) {
  if constexpr (IsFP16<kDev, T> || IsBFloat16<kDev, T>) {
    auto float_value = Caster<kDev>::template Cast<float>(value);

    return float_value <= 0.0f ? Caster<kDev>::template Cast<T>(0.0f) : value;
  } else if constexpr (std::is_unsigned_v<T>) {
    return value;
  } else {
    return value <= static_cast<T>(0) ? static_cast<T>(0) : value;
  }
}

}  // namespace

template <Device::Type kDev, typename T, unsigned int kBlockSize>
__global__ void ReluKernel(T* out, const T* input,
                           const size_t* __restrict__ out_shape,
                           const size_t* __restrict__ input_shape,
                           const ptrdiff_t* __restrict__ out_strides,
                           const ptrdiff_t* __restrict__ input_strides,
                           size_t output_size, size_t ndim, bool out_contiguous,
                           bool input_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t out_idx =
        out_contiguous ? idx : IndexToOffset(idx, ndim, out_shape, out_strides);
    size_t input_idx =
        input_contiguous ? idx
                         : IndexToOffset(idx, ndim, input_shape, input_strides);
    out[out_idx] = ReluValue<kDev>(input[input_idx]);
  }
}

}  // namespace infini::ops

#endif
