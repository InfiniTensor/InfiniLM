#ifndef INFINI_OPS_CUDA_SILU_AND_MUL_KERNEL_CUH_
#define INFINI_OPS_CUDA_SILU_AND_MUL_KERNEL_CUH_

#include <cmath>

#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

namespace detail {

// Optimized sigmoid function with support for FP16 and BF16 types.
// TODO: The unified FP16/BF16 branch uses `Caster` and scalar float
// arithmetic instead of native vectorized intrinsics (e.g. `h2rcp`,
// `__hmul2`). Profile and restore specialized paths if needed.
template <Device::Type kDev, typename T>
__device__ __forceinline__ T SiluAndMulSigmoid(const T& x) {
  if constexpr (IsFP16<kDev, T> || IsBFloat16<kDev, T>) {
    float xf = Caster<kDev>::template Cast<float>(x);
    return Caster<kDev>::template Cast<T>(
        __frcp_rn(__fadd_rn(1.0f, __expf(-xf))));
  } else if constexpr (std::is_same_v<T, float>) {
    return __frcp_rn(__fadd_rn(1.0f, __expf(-x)));
  } else {
    return 1.0f / (1.0f + expf(-x));
  }
}

}  // namespace detail

template <Device::Type kDev, typename T, unsigned int BLOCK_SIZE>
__global__ void SiluAndMulKernel(T* __restrict__ out,
                                 const T* __restrict__ input,
                                 const size_t* __restrict__ out_shape,
                                 const size_t* __restrict__ input_shape,
                                 const ptrdiff_t* __restrict__ out_strides,
                                 const ptrdiff_t* __restrict__ input_strides,
                                 size_t output_size, size_t ndim,
                                 size_t hidden_size, bool out_contiguous,
                                 bool input_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t out_idx;

    if (out_contiguous) {
      out_idx = idx;
    } else {
      out_idx = IndexToOffset(idx, ndim, out_shape, out_strides);
    }

    const size_t row = idx / hidden_size;
    const size_t column = idx % hidden_size;
    const size_t gate_logical_idx = row * 2 * hidden_size + column;
    const size_t up_logical_idx = gate_logical_idx + hidden_size;
    const size_t gate_idx =
        input_contiguous
            ? gate_logical_idx
            : IndexToOffset(gate_logical_idx, ndim, input_shape, input_strides);
    const size_t up_idx =
        input_contiguous
            ? up_logical_idx
            : IndexToOffset(up_logical_idx, ndim, input_shape, input_strides);

    T gate = input[gate_idx];
    T up = input[up_idx];

    if constexpr (IsFP16<kDev, T> || IsBFloat16<kDev, T>) {
      float gatef = Caster<kDev>::template Cast<float>(gate);
      float upf = Caster<kDev>::template Cast<float>(up);
      float sigf = __frcp_rn(__fadd_rn(1.0f, __expf(-gatef)));
      out[out_idx] = Caster<kDev>::template Cast<T>(
          __fmul_rn(__fmul_rn(gatef, sigf), upf));
    } else if constexpr (std::is_same_v<T, float>) {
      out[out_idx] =
          __fmul_rn(__fmul_rn(gate, detail::SiluAndMulSigmoid<kDev>(gate)), up);
    } else {
      out[out_idx] = gate * detail::SiluAndMulSigmoid<kDev>(gate) * up;
    }
  }
}

}  // namespace infini::ops

#endif
