#ifndef INFINI_OPS_CUDA_FUSED_ADD_RMS_NORM_KERNEL_CUH_
#define INFINI_OPS_CUDA_FUSED_ADD_RMS_NORM_KERNEL_CUH_

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {
namespace fused_add_rms_norm_detail {

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

}  // namespace fused_add_rms_norm_detail

template <unsigned int block_size, Device::Type kDev, typename TCompute,
          typename TData>
__global__ void FusedAddRmsNormKernel(TData* input, int64_t stride_input,
                                      TData* residual, int64_t stride_residual,
                                      const TData* __restrict__ weight,
                                      size_t dim, float epsilon) {
  auto input_ptr = input + blockIdx.x * stride_input;
  auto residual_ptr = residual + blockIdx.x * stride_residual;

  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    TCompute sum_val = Caster<kDev>::template Cast<TCompute>(input_ptr[i]) +
                       Caster<kDev>::template Cast<TCompute>(residual_ptr[i]);
    residual_ptr[i] = Caster<kDev>::template Cast<TData>(sum_val);
  }

  TCompute sum_squared =
      fused_add_rms_norm_detail::SumSquared<block_size, kDev, TData, TCompute>(
          residual_ptr, dim);

  __shared__ TCompute rms;
  if (threadIdx.x == 0) {
    rms = Caster<kDev>::template Cast<TCompute>(rsqrtf(
        sum_squared / Caster<kDev>::template Cast<TCompute>(dim) + epsilon));
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < dim; i += block_size) {
    TCompute value = Caster<kDev>::template Cast<TCompute>(residual_ptr[i]);
    value *= rms;
    if (weight != nullptr) {
      value *= Caster<kDev>::template Cast<TCompute>(weight[i]);
    }
    input_ptr[i] = Caster<kDev>::template Cast<TData>(value);
  }
}

}  // namespace infini::ops

#endif
