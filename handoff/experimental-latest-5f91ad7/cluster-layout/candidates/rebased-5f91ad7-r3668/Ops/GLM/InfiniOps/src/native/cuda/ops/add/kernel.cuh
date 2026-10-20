#ifndef INFINI_OPS_CUDA_ADD_KERNEL_CUH_
#define INFINI_OPS_CUDA_ADD_KERNEL_CUH_

#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

template <Device::Type kDev>
struct AddOp {
  static constexpr std::size_t num_inputs = 2;

  double alpha;

  template <typename T>
  __device__ __forceinline__ T operator()(const T& input,
                                          const T& other) const {
    if constexpr (IsFP16<kDev, T> || IsBFloat16<kDev, T>) {
      auto input_float = Caster<kDev>::template Cast<float>(input);
      auto other_float = Caster<kDev>::template Cast<float>(other);
      return Caster<kDev>::template Cast<T>(
          input_float + static_cast<float>(alpha) * other_float);
    } else if constexpr (std::is_same_v<T, float>) {
      return __fadd_rn(input, __fmul_rn(static_cast<float>(alpha), other));
    } else {
      return input + static_cast<T>(alpha) * other;
    }
  }
};

template <Device::Type kDev, typename T, unsigned int BLOCK_SIZE>
__global__ void AddKernel(T* __restrict__ out, const T* __restrict__ input,
                          const T* __restrict__ other,
                          const size_t* __restrict__ out_shape,
                          const size_t* __restrict__ input_shape,
                          const size_t* __restrict__ other_shape,
                          const ptrdiff_t* __restrict__ out_strides,
                          const ptrdiff_t* __restrict__ input_strides,
                          const ptrdiff_t* __restrict__ other_strides,
                          size_t output_size, size_t ndim, bool out_contiguous,
                          bool input_contiguous, bool other_contiguous,
                          double alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t out_idx =
        out_contiguous ? idx : IndexToOffset(idx, ndim, out_shape, out_strides);
    size_t input_idx =
        input_contiguous ? idx
                         : IndexToOffset(idx, ndim, input_shape, input_strides);
    size_t other_idx =
        other_contiguous ? idx
                         : IndexToOffset(idx, ndim, other_shape, other_strides);

    out[out_idx] = AddOp<kDev>{alpha}(input[input_idx], other[other_idx]);
  }
}

}  // namespace infini::ops

#endif
