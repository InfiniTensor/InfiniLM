#ifndef INFINI_OPS_CUDA_FILL_KERNEL_CUH_
#define INFINI_OPS_CUDA_FILL_KERNEL_CUH_

#include <cstddef>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

template <Device::Type kDev, typename T, unsigned int block_size>
__global__ void FillKernel(T* __restrict__ out,
                           const size_t* __restrict__ out_shape,
                           const ptrdiff_t* __restrict__ out_strides,
                           size_t output_size, size_t ndim, bool out_contiguous,
                           double value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t out_idx =
        out_contiguous ? idx : IndexToOffset(idx, ndim, out_shape, out_strides);
    out[out_idx] = Caster<kDev>::template Cast<T>(value);
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_FILL_KERNEL_CUH_
