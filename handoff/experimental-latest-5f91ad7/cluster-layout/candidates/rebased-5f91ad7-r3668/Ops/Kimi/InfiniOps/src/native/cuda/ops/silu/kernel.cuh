#ifndef INFINI_OPS_CUDA_SILU_KERNEL_CUH_
#define INFINI_OPS_CUDA_SILU_KERNEL_CUH_

#include <cmath>

#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {
namespace silu_detail {

// Same semantics as `third_party/InfiniCore/.../silu/cuda/kernel.cuh::SiluOp`.
template <Device::Type kDev, typename T>
__device__ __forceinline__ T Silu(const T& x) {
  if constexpr (IsFP16<kDev, T> || IsBFloat16<kDev, T>) {
    float xf = Caster<kDev>::template Cast<float>(x);
    float sigf = __frcp_rn(__fadd_rn(1.0f, __expf(-xf)));
    return Caster<kDev>::template Cast<T>(__fmul_rn(xf, sigf));
  } else if constexpr (std::is_same_v<T, float>) {
    return __fmul_rn(x, __frcp_rn(__fadd_rn(1.0f, __expf(-x))));
  } else {
    return x / (T{1} + exp(-x));
  }
}

}  // namespace silu_detail

template <Device::Type kDev, typename T, unsigned int BLOCK_SIZE>
__global__ void SiluKernel(T* __restrict__ out, const T* __restrict__ input,
                           const size_t* __restrict__ out_shape,
                           const size_t* __restrict__ input_shape,
                           const ptrdiff_t* __restrict__ out_strides,
                           const ptrdiff_t* __restrict__ input_strides,
                           size_t output_size, size_t ndim, bool out_contiguous,
                           bool input_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= output_size) {
    return;
  }

  size_t out_idx =
      out_contiguous ? idx : IndexToOffset(idx, ndim, out_shape, out_strides);
  size_t input_idx = input_contiguous
                         ? idx
                         : IndexToOffset(idx, ndim, input_shape, input_strides);

  out[out_idx] = silu_detail::Silu<kDev>(input[input_idx]);
}

}  // namespace infini::ops

#endif
