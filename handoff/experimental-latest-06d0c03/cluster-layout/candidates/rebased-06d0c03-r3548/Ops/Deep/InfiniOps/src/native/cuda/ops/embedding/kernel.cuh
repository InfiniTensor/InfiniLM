#ifndef INFINI_OPS_CUDA_EMBEDDING_KERNEL_CUH_
#define INFINI_OPS_CUDA_EMBEDDING_KERNEL_CUH_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <type_traits>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {
namespace embedding_detail {

struct MaxOp {
  __device__ float operator()(float lhs, float rhs) const {
    return lhs > rhs ? lhs : rhs;
  }
};

struct MinOp {
  __device__ float operator()(float lhs, float rhs) const {
    return lhs < rhs ? lhs : rhs;
  }
};

template <unsigned int block_size>
__device__ float ReduceSum(float value) {
  using BlockReduce = cub::BlockReduce<float, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Sum(value);
}

template <unsigned int block_size>
__device__ float ReduceMax(float value) {
  using BlockReduce = cub::BlockReduce<float, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Reduce(value, MaxOp{});
}

template <unsigned int block_size>
__device__ float ReduceMin(float value) {
  using BlockReduce = cub::BlockReduce<float, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  return BlockReduce(temp_storage).Reduce(value, MinOp{});
}

__forceinline__ __device__ bool IsAligned(const void* ptr, size_t alignment) {
  return (reinterpret_cast<size_t>(ptr) % alignment) == 0;
}

// MUSA `__ldg` does not provide overloads for 16-bit float types such as
// `__mt_bfloat16`, so fall back to a plain global load on Moore.
template <Device::Type kDev, typename T>
__forceinline__ __device__ T LoadGlobal(const T* ptr) {
  if constexpr (kDev == Device::Type::kMoore &&
                (IsFP16<kDev, T> || IsBFloat16<kDev, T>)) {
    return *ptr;
  } else {
    return __ldg(ptr);
  }
}

template <Device::Type kDev, typename T>
__forceinline__ __device__ void CopyScalar(T* __restrict__ dst,
                                           const T* __restrict__ src,
                                           size_t embedding_dim,
                                           ptrdiff_t dst_col_stride = 1,
                                           ptrdiff_t src_col_stride = 1) {
  for (size_t i = 0; i < embedding_dim; ++i) {
    dst[i * dst_col_stride] = LoadGlobal<kDev, T>(&src[i * src_col_stride]);
  }
}

// Same as `third_party/InfiniCore/.../embedding/cuda/embedding_kernel.cuh`.
template <typename T>
__forceinline__ __device__ void CopyVectorizedFloat4(
    float* __restrict__ dst, const float* __restrict__ src,
    size_t embedding_dim) {
  const float4* src_vec = reinterpret_cast<const float4*>(src);
  float4* dst_vec = reinterpret_cast<float4*>(dst);
  size_t vec_count = embedding_dim / 4;

  for (size_t i = 0; i < vec_count; ++i) {
    dst_vec[i] = __ldg(&src_vec[i]);
  }

  size_t remaining = embedding_dim % 4;
  if (remaining > 0) {
    size_t offset = vec_count * 4;
    for (size_t i = 0; i < remaining; ++i) {
      dst[offset + i] = __ldg(&src[offset + i]);
    }
  }
}

template <typename T>
__forceinline__ __device__ void CopyVectorizedFloat2(
    float* __restrict__ dst, const float* __restrict__ src,
    size_t embedding_dim) {
  const float2* src_vec = reinterpret_cast<const float2*>(src);
  float2* dst_vec = reinterpret_cast<float2*>(dst);
  size_t vec_count = embedding_dim / 2;

  for (size_t i = 0; i < vec_count; ++i) {
    dst_vec[i] = __ldg(&src_vec[i]);
  }

  if (embedding_dim % 2 != 0) {
    dst[embedding_dim - 1] = __ldg(&src[embedding_dim - 1]);
  }
}

template <Device::Type kDev, typename T>
__forceinline__ __device__ void CopyVectorized16(T* __restrict__ dst,
                                                 const T* __restrict__ src,
                                                 size_t embedding_dim) {
  const uint32_t* src_vec = reinterpret_cast<const uint32_t*>(src);
  uint32_t* dst_vec = reinterpret_cast<uint32_t*>(dst);
  size_t vec_count = embedding_dim / 2;

  for (size_t i = 0; i < vec_count; ++i) {
    dst_vec[i] = __ldg(&src_vec[i]);
  }

  if (embedding_dim % 2 != 0) {
    dst[embedding_dim - 1] = LoadGlobal<kDev, T>(&src[embedding_dim - 1]);
  }
}

// Contiguous row copy with InfiniCore vectorization strategy.
template <Device::Type kDev, typename T>
__forceinline__ __device__ void CopyRowContiguous(T* __restrict__ dst,
                                                  const T* __restrict__ src,
                                                  size_t embedding_dim) {
  if constexpr (std::is_same_v<T, float>) {
    bool aligned_16 = IsAligned(src, 16) && IsAligned(dst, 16);
    if (aligned_16 && embedding_dim >= 4 && embedding_dim % 4 == 0) {
      CopyVectorizedFloat4<T>(dst, src, embedding_dim);
    } else if (embedding_dim >= 2 && embedding_dim % 2 == 0 &&
               IsAligned(src, alignof(float2)) &&
               IsAligned(dst, alignof(float2))) {
      CopyVectorizedFloat2<T>(dst, src, embedding_dim);
    } else {
      CopyScalar<kDev, T>(dst, src, embedding_dim);
    }
  } else if constexpr (IsFP16<kDev, T> || IsBFloat16<kDev, T>) {
    if (embedding_dim >= 2 && embedding_dim % 2 == 0 &&
        IsAligned(src, alignof(uint32_t)) &&
        IsAligned(dst, alignof(uint32_t))) {
      CopyVectorized16<kDev, T>(dst, src, embedding_dim);
    } else {
      CopyScalar<kDev, T>(dst, src, embedding_dim);
    }
  } else {
    CopyScalar<kDev, T>(dst, src, embedding_dim);
  }
}

template <Device::Type kDev, typename T>
__forceinline__ __device__ void CopyRow(T* __restrict__ dst,
                                        const T* __restrict__ src,
                                        size_t embedding_dim,
                                        ptrdiff_t dst_col_stride,
                                        ptrdiff_t src_col_stride) {
  if (dst_col_stride == 1 && src_col_stride == 1) {
    CopyRowContiguous<kDev>(dst, src, embedding_dim);
    return;
  }

  CopyScalar<kDev, T>(dst, src, embedding_dim, dst_col_stride, src_col_stride);
}

}  // namespace embedding_detail

template <unsigned int block_size>
__global__ void ClearEmbeddingVisitedKernel(int* visited, size_t vocab_size) {
  const size_t index = blockIdx.x * block_size + threadIdx.x;

  if (index < vocab_size) {
    visited[index] = 0;
  }
}

template <unsigned int block_size, Device::Type kDev, typename T,
          typename IndexT>
__global__ void EmbeddingRenormKernel(
    T* weight, const IndexT* indices, int* visited, size_t num_indices,
    size_t input_ndim, const size_t* input_shape,
    const ptrdiff_t* input_strides, ptrdiff_t weight_row_stride,
    ptrdiff_t weight_col_stride, size_t embedding_dim, size_t vocab_size,
    bool input_contiguous, double max_norm, double norm_type) {
  const size_t flat_index = blockIdx.x;

  if (flat_index >= num_indices) {
    return;
  }

  __shared__ size_t row;
  __shared__ bool claimed;

  if (threadIdx.x == 0) {
    const size_t input_offset =
        input_contiguous
            ? flat_index
            : IndexToOffset(flat_index, input_ndim, input_shape, input_strides);
    const IndexT index = indices[input_offset];
    claimed = index >= 0 && static_cast<size_t>(index) < vocab_size;

    if (claimed) {
      row = static_cast<size_t>(index);
      claimed = atomicCAS(visited + row, 0, 1) == 0;
    }
  }

  __syncthreads();

  if (!claimed) {
    return;
  }

  const float p = static_cast<float>(norm_type);
  float norm;

  if (isinf(p) && p > 0.0f) {
    float local_max = 0.0f;
    for (size_t column = threadIdx.x; column < embedding_dim;
         column += block_size) {
      const float value = Caster<kDev>::template Cast<float>(
          weight[row * weight_row_stride + column * weight_col_stride]);
      local_max = fmaxf(local_max, fabsf(value));
    }
    norm = embedding_detail::ReduceMax<block_size>(local_max);
  } else if (isinf(p)) {
    float local_min = INFINITY;
    for (size_t column = threadIdx.x; column < embedding_dim;
         column += block_size) {
      const float value = Caster<kDev>::template Cast<float>(
          weight[row * weight_row_stride + column * weight_col_stride]);
      local_min = fminf(local_min, fabsf(value));
    }
    norm = embedding_detail::ReduceMin<block_size>(local_min);
  } else {
    float local_sum = 0.0f;
    for (size_t column = threadIdx.x; column < embedding_dim;
         column += block_size) {
      const float value = Caster<kDev>::template Cast<float>(
          weight[row * weight_row_stride + column * weight_col_stride]);
      const float abs_value = fabsf(value);

      if (p == 0.0f) {
        local_sum += abs_value == 0.0f ? 0.0f : 1.0f;
      } else if (p == 1.0f) {
        local_sum += abs_value;
      } else if (p == 2.0f) {
        local_sum += value * value;
      } else {
        local_sum += powf(abs_value, p);
      }
    }

    const float sum = embedding_detail::ReduceSum<block_size>(local_sum);
    norm = p == 0.0f ? sum : powf(sum, 1.0f / p);
  }

  __shared__ float factor;
  if (threadIdx.x == 0) {
    const float limit = static_cast<float>(max_norm);
    factor = norm > limit ? limit / (norm + 1e-7f) : 1.0f;
  }

  __syncthreads();

  if (factor == 1.0f) {
    return;
  }

  for (size_t column = threadIdx.x; column < embedding_dim;
       column += block_size) {
    const auto offset = row * weight_row_stride + column * weight_col_stride;
    const float current = Caster<kDev>::template Cast<float>(weight[offset]);
    weight[offset] = Caster<kDev>::template Cast<T>(current * factor);
  }
}

template <Device::Type kDev, typename T, typename IndexT>
__global__ void EmbeddingKernel(
    T* __restrict__ output, const IndexT* __restrict__ indices,
    const T* __restrict__ weight, size_t num_indices, size_t input_ndim,
    const size_t* __restrict__ input_shape,
    const ptrdiff_t* __restrict__ input_strides, size_t out_ndim,
    const size_t* __restrict__ out_shape,
    const ptrdiff_t* __restrict__ out_strides, ptrdiff_t weight_row_stride,
    ptrdiff_t weight_col_stride, size_t embedding_dim, size_t vocab_size,
    bool input_contiguous, bool out_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_indices) {
    return;
  }

  size_t input_offset =
      input_contiguous
          ? idx
          : IndexToOffset(idx, input_ndim, input_shape, input_strides);
  IndexT index_val = __ldg(&indices[input_offset]);

  if (index_val < 0 || static_cast<size_t>(index_val) >= vocab_size) {
    return;
  }

  const T* src = weight + static_cast<size_t>(index_val) * weight_row_stride;

  if (out_contiguous) {
    T* dst = output + idx * embedding_dim;
    embedding_detail::CopyRow<kDev>(dst, src, embedding_dim, 1,
                                    weight_col_stride);
    return;
  }

  size_t out_prefix_ndim = out_ndim > 0 ? out_ndim - 1 : 0;
  size_t out_row_offset =
      IndexToOffset(idx, out_prefix_ndim, out_shape, out_strides);
  ptrdiff_t out_col_stride = out_strides[out_ndim - 1];

  embedding_detail::CopyRow<kDev>(output + out_row_offset, src, embedding_dim,
                                  out_col_stride, weight_col_stride);
}

}  // namespace infini::ops

#endif
