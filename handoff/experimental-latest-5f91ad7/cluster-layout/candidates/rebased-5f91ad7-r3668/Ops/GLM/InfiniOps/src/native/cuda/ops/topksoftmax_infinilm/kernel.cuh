#ifndef INFINI_OPS_CUDA_TOPKSOFTMAX_INFINILM_KERNEL_CUH_
#define INFINI_OPS_CUDA_TOPKSOFTMAX_INFINILM_KERNEL_CUH_

#include <cfloat>
#include <cmath>
#include <cstddef>

#include "native/cuda/caster.cuh"

namespace infini::ops {

namespace {

template <unsigned int block_size>
__device__ __forceinline__ float TopksoftmaxInfinilmBlockMax(float value) {
  __shared__ float values[block_size];
  values[threadIdx.x] = value;
  __syncthreads();

  for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      float other = values[threadIdx.x + stride];
      values[threadIdx.x] =
          values[threadIdx.x] > other ? values[threadIdx.x] : other;
    }
    __syncthreads();
  }

  return values[0];
}

template <unsigned int block_size>
__device__ __forceinline__ float TopksoftmaxInfinilmBlockSum(float value) {
  __shared__ float values[block_size];
  values[threadIdx.x] = value;
  __syncthreads();

  for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      values[threadIdx.x] += values[threadIdx.x + stride];
    }
    __syncthreads();
  }

  return values[0];
}

__device__ __forceinline__ bool TopksoftmaxInfinilmBetter(float value,
                                                          int index,
                                                          float best_value,
                                                          int best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

template <unsigned int block_size>
__device__ __forceinline__ void TopksoftmaxInfinilmBlockBest(float& value,
                                                             int& index) {
  __shared__ float values[block_size];
  __shared__ int indices[block_size];
  values[threadIdx.x] = value;
  indices[threadIdx.x] = index;
  __syncthreads();

  for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      float other_value = values[threadIdx.x + stride];
      int other_index = indices[threadIdx.x + stride];
      if (TopksoftmaxInfinilmBetter(other_value, other_index,
                                    values[threadIdx.x],
                                    indices[threadIdx.x])) {
        values[threadIdx.x] = other_value;
        indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }

  value = values[0];
  index = indices[0];
}

}  // namespace

template <unsigned int block_size, Device::Type kDev, typename T>
__global__ void TopksoftmaxInfinilmKernel(
    float* __restrict__ values, int32_t* __restrict__ indices,
    const T* __restrict__ input, const ptrdiff_t* __restrict__ input_strides,
    const ptrdiff_t* __restrict__ values_strides,
    const ptrdiff_t* __restrict__ indices_strides, size_t row_count,
    size_t width, size_t topk, bool norm) {
  size_t row = blockIdx.x + blockIdx.y * gridDim.x;
  if (row >= row_count) {
    return;
  }

  ptrdiff_t input_base = row * input_strides[0];
  ptrdiff_t values_base = row * values_strides[0];
  ptrdiff_t indices_base = row * indices_strides[0];

  float thread_max = -FLT_MAX;
  for (size_t col = threadIdx.x; col < width; col += block_size) {
    float value = Caster<kDev>::template Cast<float>(
        input[input_base + col * input_strides[1]]);
    thread_max = thread_max > value ? thread_max : value;
  }

  float max_value = TopksoftmaxInfinilmBlockMax<block_size>(thread_max);

  float thread_sum = 0.0f;
  for (size_t col = threadIdx.x; col < width; col += block_size) {
    float value = Caster<kDev>::template Cast<float>(
        input[input_base + col * input_strides[1]]);
    thread_sum += expf(value - max_value);
  }

  float softmax_sum = TopksoftmaxInfinilmBlockSum<block_size>(thread_sum);

  for (size_t rank = 0; rank < topk; ++rank) {
    float thread_best = -FLT_MAX;
    int thread_index = -1;

    for (size_t col = threadIdx.x; col < width; col += block_size) {
      bool selected = false;
      for (size_t prev = 0; prev < rank; ++prev) {
        if (indices[indices_base + prev * indices_strides[1]] ==
            static_cast<int32_t>(col)) {
          selected = true;
          break;
        }
      }

      if (!selected) {
        float value = Caster<kDev>::template Cast<float>(
            input[input_base + col * input_strides[1]]);
        float softmax_value = expf(value - max_value) / softmax_sum;
        if (TopksoftmaxInfinilmBetter(softmax_value, static_cast<int>(col),
                                      thread_best, thread_index)) {
          thread_best = softmax_value;
          thread_index = static_cast<int>(col);
        }
      }
    }

    TopksoftmaxInfinilmBlockBest<block_size>(thread_best, thread_index);

    if (threadIdx.x == 0) {
      values[values_base + rank * values_strides[1]] = thread_best;
      indices[indices_base + rank * indices_strides[1]] = thread_index;
    }
    __syncthreads();
  }

  if (norm) {
    float thread_topk_sum = 0.0f;
    for (size_t rank = threadIdx.x; rank < topk; rank += block_size) {
      thread_topk_sum += values[values_base + rank * values_strides[1]];
    }
    float topk_sum = TopksoftmaxInfinilmBlockSum<block_size>(thread_topk_sum);

    for (size_t rank = threadIdx.x; rank < topk; rank += block_size) {
      ptrdiff_t offset = values_base + rank * values_strides[1];
      values[offset] = values[offset] / topk_sum;
    }
  }
}

}  // namespace infini::ops

#endif
