#ifndef INFINI_OPS_CUDA_SIGMOID_INFINILM_KERNEL_CUH_
#define INFINI_OPS_CUDA_SIGMOID_INFINILM_KERNEL_CUH_

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

namespace {

template <Device::Type kDev, typename T>
__device__ __forceinline__ T SigmoidInfinilmValue(T x) {
  if constexpr (std::is_same_v<T, double>) {
    return 1.0 / (1.0 + exp(-x));
  } else {
    const float v = Caster<kDev>::template Cast<float>(x);
    const float y = 1.0f / (1.0f + expf(-v));
    return Caster<kDev>::template Cast<T>(y);
  }
}

}  // namespace

template <Device::Type kDev, typename T, unsigned int block_size>
__global__ void SigmoidInfinilmKernel(
    T* __restrict__ out, const T* __restrict__ input,
    const size_t* __restrict__ out_shape,
    const size_t* __restrict__ input_shape,
    const ptrdiff_t* __restrict__ out_strides,
    const ptrdiff_t* __restrict__ input_strides, size_t output_size,
    size_t ndim, bool out_contiguous, bool input_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t out_idx =
        out_contiguous ? idx : IndexToOffset(idx, ndim, out_shape, out_strides);
    size_t input_idx =
        input_contiguous ? idx
                         : IndexToOffset(idx, ndim, input_shape, input_strides);
    out[out_idx] = SigmoidInfinilmValue<kDev>(input[input_idx]);
  }
}

}  // namespace infini::ops

#endif
