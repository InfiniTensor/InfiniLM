#ifndef INFINI_OPS_CUDA_SOFTMAX_INFINILM_KERNEL_CUH_
#define INFINI_OPS_CUDA_SOFTMAX_INFINILM_KERNEL_CUH_

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cub/block/block_reduce.cuh>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

namespace {

struct SoftmaxInfinilmMaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a > b ? a : b;
  }
};

template <unsigned int block_size>
__device__ __forceinline__ float BlockMax(float value) {
  using BlockReduce = cub::BlockReduce<float, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Reduce(value, SoftmaxInfinilmMaxOp());
}

template <unsigned int block_size>
__device__ __forceinline__ float BlockSum(float value) {
  using BlockReduce = cub::BlockReduce<float, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Sum(value);
}

__device__ __forceinline__ size_t SoftmaxInfinilmRowOffset(
    size_t row, size_t ndim, size_t dim, const size_t* __restrict__ shape,
    const ptrdiff_t* __restrict__ strides) {
  size_t offset = 0;
  for (size_t axis = ndim; axis > 0; --axis) {
    size_t i = axis - 1;
    if (i == dim) {
      continue;
    }
    size_t coord = row % shape[i];
    row /= shape[i];
    offset += coord * strides[i];
  }
  return offset;
}

}  // namespace

template <unsigned int block_size, Device::Type kDev, typename T>
__global__ void SoftmaxInfinilmKernel(
    T* __restrict__ out, const T* __restrict__ input,
    const size_t* __restrict__ shape, const ptrdiff_t* __restrict__ out_strides,
    const ptrdiff_t* __restrict__ input_strides, size_t row_count,
    size_t dim_size, size_t ndim, size_t dim) {
  size_t row = blockIdx.x + blockIdx.y * gridDim.x;
  if (row >= row_count) {
    return;
  }

  size_t input_base =
      SoftmaxInfinilmRowOffset(row, ndim, dim, shape, input_strides);
  size_t out_base =
      SoftmaxInfinilmRowOffset(row, ndim, dim, shape, out_strides);
  ptrdiff_t input_dim_stride = input_strides[dim];
  ptrdiff_t out_dim_stride = out_strides[dim];

  float thread_max = -FLT_MAX;
  for (size_t i = threadIdx.x; i < dim_size; i += block_size) {
    float value = Caster<kDev>::template Cast<float>(
        input[input_base + i * input_dim_stride]);
    thread_max = thread_max > value ? thread_max : value;
  }

  float block_max = BlockMax<block_size>(thread_max);
  __shared__ float max_value;
  if (threadIdx.x == 0) {
    max_value = block_max;
  }
  __syncthreads();

  float thread_sum = 0.0f;
  for (size_t i = threadIdx.x; i < dim_size; i += block_size) {
    float value = Caster<kDev>::template Cast<float>(
        input[input_base + i * input_dim_stride]);
    float exp_value = expf(value - max_value);
    thread_sum += exp_value;
    out[out_base + i * out_dim_stride] =
        Caster<kDev>::template Cast<T>(exp_value);
  }

  float block_sum = BlockSum<block_size>(thread_sum);
  __shared__ float sum_value;
  if (threadIdx.x == 0) {
    sum_value = block_sum;
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < dim_size; i += block_size) {
    float value =
        Caster<kDev>::template Cast<float>(out[out_base + i * out_dim_stride]);
    out[out_base + i * out_dim_stride] =
        Caster<kDev>::template Cast<T>(value / sum_value);
  }
}

}  // namespace infini::ops

#endif
