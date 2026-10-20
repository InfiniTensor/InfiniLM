#ifndef INFINI_OPS_CUDA_CAUSAL_SOFTMAX_KERNEL_CUH_
#define INFINI_OPS_CUDA_CAUSAL_SOFTMAX_KERNEL_CUH_

#include <cmath>
#include <cstddef>
#include <cub/block/block_reduce.cuh>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

namespace {

template <Device::Type kDev, typename Data, typename Compute>
__device__ __forceinline__ Data ExpAndCast(Compute x) {
  return Caster<kDev>::template Cast<Data>(
      expf(Caster<kDev>::template Cast<float>(x)));
}

struct BlockMaxOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return (a > b) ? a : b;
  }
};

template <unsigned int block_size, typename Data>
__device__ __forceinline__ Data BlockMax(const Data* data_ptr, size_t count) {
  Data thread_max = count > 0 ? data_ptr[0] : Data{};
  for (size_t i = threadIdx.x; i < count; i += block_size) {
    Data v = data_ptr[i];
    thread_max = (v > thread_max) ? v : thread_max;
  }
  using BlockReduce = cub::BlockReduce<Data, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Reduce(thread_max, BlockMaxOp());
}

template <unsigned int block_size, Device::Type kDev, typename Data,
          typename Compute>
__device__ __forceinline__ Compute BlockSum(const Data* data_ptr,
                                            size_t count) {
  Compute thread_sum = 0;
  for (size_t i = threadIdx.x; i < count; i += block_size) {
    thread_sum += Caster<kDev>::template Cast<Compute>(data_ptr[i]);
  }
  using BlockReduce = cub::BlockReduce<Compute, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Sum(thread_sum);
}

}  // namespace

template <unsigned int block_size, Device::Type kDev, typename Data,
          typename Compute>
__global__ void CausalSoftmaxKernel(
    Data* __restrict__ out_ptr, const Data* __restrict__ input_ptr,
    size_t batch_size, size_t seq_len, size_t total_seq_len,
    int64_t stride_out_batch, int64_t stride_out_row,
    int64_t stride_input_batch, int64_t stride_input_row) {
  size_t row_idx = blockIdx.x;
  size_t batch_idx = blockIdx.y;

  Data* out_row =
      out_ptr + batch_idx * stride_out_batch + row_idx * stride_out_row;
  const Data* input_row =
      input_ptr + batch_idx * stride_input_batch + row_idx * stride_input_row;

  size_t valid_len = total_seq_len - seq_len + row_idx + 1;

  __shared__ Data max_val;
  Data block_max = BlockMax<block_size, Data>(input_row, valid_len);
  if (threadIdx.x == 0) {
    max_val = block_max;
  }
  __syncthreads();

  for (size_t col = threadIdx.x; col < total_seq_len; col += block_size) {
    if (col < valid_len) {
      Compute diff = Caster<kDev>::template Cast<Compute>(input_row[col]) -
                     Caster<kDev>::template Cast<Compute>(max_val);
      out_row[col] = ExpAndCast<kDev, Data, Compute>(diff);
    } else {
      out_row[col] = Caster<kDev>::template Cast<Data>(0.0f);
    }
  }
  __syncthreads();

  __shared__ Compute sum_val;
  Compute block_sum =
      BlockSum<block_size, kDev, Data, Compute>(out_row, total_seq_len);
  if (threadIdx.x == 0) {
    sum_val = block_sum;
  }
  __syncthreads();

  for (size_t col = threadIdx.x; col < total_seq_len; col += block_size) {
    Compute quot = Caster<kDev>::template Cast<Compute>(out_row[col]) / sum_val;
    out_row[col] = Caster<kDev>::template Cast<Data>(quot);
  }
}

}  // namespace infini::ops

#endif
