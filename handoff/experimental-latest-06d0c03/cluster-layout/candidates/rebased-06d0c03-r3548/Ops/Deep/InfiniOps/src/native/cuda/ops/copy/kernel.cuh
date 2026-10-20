#ifndef INFINI_OPS_CUDA_COPY_KERNEL_CUH_
#define INFINI_OPS_CUDA_COPY_KERNEL_CUH_

#include <cstddef>

#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

template <typename T, unsigned int block_size>
__global__ void CopyKernel(T* __restrict__ out, const T* __restrict__ input,
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
    out[out_idx] = input[input_idx];
  }
}

}  // namespace infini::ops

#endif
