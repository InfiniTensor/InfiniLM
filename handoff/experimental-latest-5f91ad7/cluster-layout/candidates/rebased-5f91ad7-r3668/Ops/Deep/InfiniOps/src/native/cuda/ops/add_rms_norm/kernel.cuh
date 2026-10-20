#ifndef INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_CUH_
#define INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_CUH_

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {
namespace add_rms_norm_detail {

// Same as `native/cuda/ops/rms_norm/kernel.cuh`.
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

}  // namespace add_rms_norm_detail

template <unsigned int block_size, Device::Type kDev, typename TCompute,
          typename TData, typename TWeight>
__global__ void AddRmsNormKernel(
    TData* __restrict__ y, int64_t stride_y_batch, int64_t stride_y_nhead,
    TData* __restrict__ residual_out, int64_t stride_residual_out_batch,
    int64_t stride_residual_out_nhead, const TData* __restrict__ input,
    int64_t stride_input_batch, int64_t stride_input_nhead,
    const TData* __restrict__ residual, int64_t stride_residual_batch,
    int64_t stride_residual_nhead, const TWeight* __restrict__ w, size_t nhead,
    size_t dim, float epsilon) {
  size_t batch_idx = blockIdx.x / nhead;
  size_t head_idx = blockIdx.x % nhead;

  auto y_ptr = y + batch_idx * stride_y_batch + head_idx * stride_y_nhead;
  auto input_ptr =
      input + batch_idx * stride_input_batch + head_idx * stride_input_nhead;
  auto residual_ptr = residual + batch_idx * stride_residual_batch +
                      head_idx * stride_residual_nhead;
  auto w_ptr = w;
  auto residual_out_ptr = residual_out + batch_idx * stride_residual_out_batch +
                          head_idx * stride_residual_out_nhead;

  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    TCompute sum_val = Caster<kDev>::template Cast<TCompute>(input_ptr[i]) +
                       Caster<kDev>::template Cast<TCompute>(residual_ptr[i]);
    residual_out_ptr[i] = Caster<kDev>::template Cast<TData>(sum_val);
  }

  TCompute sum_squared =
      add_rms_norm_detail::SumSquared<block_size, kDev, TData, TCompute>(
          residual_out_ptr, dim);

  __shared__ TCompute rms;
  if (threadIdx.x == 0) {
    rms = Caster<kDev>::template Cast<TCompute>(rsqrtf(
        sum_squared / Caster<kDev>::template Cast<TCompute>(dim) + epsilon));
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    TCompute sum_val =
        Caster<kDev>::template Cast<TCompute>(residual_out_ptr[i]);
    y_ptr[i] = Caster<kDev>::template Cast<TData>(
        sum_val * Caster<kDev>::template Cast<TCompute>(w_ptr[i]) * rms);
  }
}

}  // namespace infini::ops

#endif
