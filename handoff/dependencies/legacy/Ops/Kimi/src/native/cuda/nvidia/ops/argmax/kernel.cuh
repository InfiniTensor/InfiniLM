#ifndef INFINI_OPS_NVIDIA_ARGMAX_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_ARGMAX_KERNEL_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <climits>

namespace infini::ops::nvidia_argmax_detail {

constexpr int kThreads = 256;
constexpr int kChunkWidth = 2048;

template <typename T>
__device__ float ToFloat(T value) { return static_cast<float>(value); }

template <>
__device__ inline float ToFloat<half>(half value) { return __half2float(value); }

template <>
__device__ inline float ToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

struct Candidate {
  float value;
  int index;
};

static_assert(sizeof(Candidate) == 8);

// The frozen PyTorch Argmax contract prefers NaN over finite values and the
// first index among equal maxima (including multiple NaNs).
__device__ inline Candidate Better(Candidate a, Candidate b) {
  if (b.index == INT_MAX) return a;
  if (a.index == INT_MAX) return b;
  const bool a_nan = isnan(a.value);
  const bool b_nan = isnan(b.value);
  if (a_nan != b_nan) return b_nan ? b : a;
  if (a_nan || a.value == b.value) return b.index < a.index ? b : a;
  return b.value > a.value ? b : a;
}

template <typename T>
__global__ void RowsKernel(const T* input, std::int64_t* output, int width) {
  const int row = blockIdx.x;
  const T* values = input + static_cast<std::size_t>(row) * width;
  Candidate best{-CUDART_INF_F, INT_MAX};
  for (std::int64_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    best = Better(best, {ToFloat(values[column]), static_cast<int>(column)});
  }
  __shared__ Candidate partial[kThreads];
  partial[threadIdx.x] = best;
  __syncthreads();
  for (int offset = kThreads / 2; offset; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] =
          Better(partial[threadIdx.x], partial[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[row] = partial[0].index;
}

template <typename T>
__global__ void PartialsKernel(const T* input, Candidate* partials,
                               int rows, int width, int chunks) {
  const std::size_t tasks = static_cast<std::size_t>(rows) * chunks;
  __shared__ Candidate reduced[kThreads];
  for (std::size_t task = blockIdx.x; task < tasks; task += gridDim.x) {
    const int row = static_cast<int>(task / chunks);
    const int chunk = static_cast<int>(task % chunks);
    const int begin = chunk * kChunkWidth;
    const int end = begin + min(kChunkWidth, width - begin);
    const T* values = input + static_cast<std::size_t>(row) * width;
    Candidate best{-CUDART_INF_F, INT_MAX};
    for (std::int64_t column = std::int64_t{begin} + threadIdx.x; column < end;
         column += blockDim.x) {
      best = Better(best, {ToFloat(values[column]), static_cast<int>(column)});
    }
    reduced[threadIdx.x] = best;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset /= 2) {
      if (threadIdx.x < offset) {
        reduced[threadIdx.x] =
            Better(reduced[threadIdx.x], reduced[threadIdx.x + offset]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) partials[task] = reduced[0];
    __syncthreads();
  }
}

static __global__ void FinishKernel(const Candidate* partials,
                             std::int64_t* output, int rows, int chunks) {
  __shared__ Candidate reduced[kThreads];
  for (std::int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const Candidate* values = partials + static_cast<std::size_t>(row) * chunks;
    Candidate best{-CUDART_INF_F, INT_MAX};
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
      best = Better(best, values[chunk]);
    }
    reduced[threadIdx.x] = best;
    __syncthreads();
    for (int offset = kThreads / 2; offset; offset /= 2) {
      if (threadIdx.x < offset) {
        reduced[threadIdx.x] =
            Better(reduced[threadIdx.x], reduced[threadIdx.x + offset]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = reduced[0].index;
    __syncthreads();
  }
}

}  // namespace infini::ops::nvidia_argmax_detail

#endif  // INFINI_OPS_NVIDIA_ARGMAX_KERNEL_CUH_
