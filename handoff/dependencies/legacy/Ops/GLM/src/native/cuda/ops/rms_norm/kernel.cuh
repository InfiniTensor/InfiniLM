#ifndef INFINI_OPS_CUDA_RMS_NORM_KERNEL_CUH_
#define INFINI_OPS_CUDA_RMS_NORM_KERNEL_CUH_

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

namespace {

template <unsigned int block_size, Device::Type kDev, typename TData,
          typename TCompute>
__device__ __forceinline__ TCompute SumSquared(const TData* data_ptr,
                                               size_t count) {
  TCompute ss = 0;
  for (size_t i = threadIdx.x; i < count; i += block_size) {
    TCompute value = Caster<kDev>::template Cast<TCompute>(data_ptr[i]);
    ss += value * value;
  }
  using BlockReduce = cub::BlockReduce<TCompute, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Sum(ss);
}

}  // namespace

template <unsigned int block_size, Device::Type kDev, typename TCompute,
          typename TData, typename TWeight>
__global__ void RmsNormKernel(TData* __restrict__ y, int64_t stride_y_batch,
                              int64_t stride_y_nhead,
                              const TData* __restrict__ x,
                              int64_t stride_x_batch, int64_t stride_x_nhead,
                              const TWeight* __restrict__ w, size_t nhead,
                              size_t dim, float epsilon) {
  size_t batch_idx = blockIdx.x / nhead;
  size_t head_idx = blockIdx.x % nhead;

  auto y_ptr = y + batch_idx * stride_y_batch + head_idx * stride_y_nhead;
  auto x_ptr = x + batch_idx * stride_x_batch + head_idx * stride_x_nhead;
  auto w_ptr = w;

  TCompute ss = SumSquared<block_size, kDev, TData, TCompute>(x_ptr, dim);

  __shared__ TCompute rms;
  if (threadIdx.x == 0) {
    rms = Caster<kDev>::template Cast<TCompute>(
        rsqrtf(ss / Caster<kDev>::template Cast<TCompute>(dim) + epsilon));
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    y_ptr[i] = Caster<kDev>::template Cast<TData>(
        Caster<kDev>::template Cast<TCompute>(x_ptr[i]) *
        Caster<kDev>::template Cast<TCompute>(w_ptr[i]) * rms);
  }
}

}  // namespace infini::ops

#endif
