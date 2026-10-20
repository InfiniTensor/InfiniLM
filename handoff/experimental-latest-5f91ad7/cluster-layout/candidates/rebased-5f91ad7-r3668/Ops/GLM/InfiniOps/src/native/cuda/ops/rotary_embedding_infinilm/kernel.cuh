#ifndef INFINI_OPS_CUDA_ROPE_KERNEL_CUH_
#define INFINI_OPS_CUDA_ROPE_KERNEL_CUH_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "native/cuda/caster.cuh"

namespace infini::ops {

template <typename T>
struct VecTypeHelper {};

template <>
struct VecTypeHelper<half> {
  using type2 = half2;
  static __device__ __forceinline__ half2 make_half2(float x, float y) {
    return __floats2half2_rn(x, y);
  }
  static __device__ __forceinline__ float low(half2 v) {
    return __low2float(v);
  }
  static __device__ __forceinline__ float high(half2 v) {
    return __high2float(v);
  }
};

template <>
struct VecTypeHelper<cuda_bfloat16> {
  using type2 = cuda_bfloat162;
  static __device__ __forceinline__ cuda_bfloat162 make_half2(float x,
                                                              float y) {
    return __floats2bfloat162_rn(x, y);
  }
  static __device__ __forceinline__ float low(cuda_bfloat162 v) {
    return __low2float(v);
  }
  static __device__ __forceinline__ float high(cuda_bfloat162 v) {
    return __high2float(v);
  }
};

template <bool IsNeox, Device::Type kDev, typename TData, typename TIndex,
          typename TAngle>
__global__ void RopeKernel(
    TData* __restrict__ out_ptr, const TData* __restrict__ input_ptr,
    const TIndex* __restrict__ pos_ids_ptr, const TAngle* __restrict__ sin_ptr,
    const TAngle* __restrict__ cos_ptr, size_t table_dim,
    ptrdiff_t out_stride_batch, ptrdiff_t out_stride_seqlen,
    ptrdiff_t out_stride_nhead, ptrdiff_t input_stride_batch,
    ptrdiff_t input_stride_seqlen, ptrdiff_t input_stride_nhead,
    ptrdiff_t pos_stride_batch, bool pos_has_batch_dim, bool has_batch_dim) {
  const size_t batch_idx = has_batch_dim ? blockIdx.z : 0;
  const size_t seq_idx = blockIdx.x;
  const size_t head_idx = blockIdx.y;

  auto out_offset = (has_batch_dim ? batch_idx * out_stride_batch : 0) +
                    seq_idx * out_stride_seqlen + head_idx * out_stride_nhead;
  auto input_offset = (has_batch_dim ? batch_idx * input_stride_batch : 0) +
                      seq_idx * input_stride_seqlen +
                      head_idx * input_stride_nhead;

  size_t pos_offset;
  if (pos_has_batch_dim) {
    pos_offset = batch_idx * pos_stride_batch + seq_idx;
  } else {
    pos_offset = seq_idx;
  }

  size_t pos_id = static_cast<size_t>(pos_ids_ptr[pos_offset]);
  size_t table_offset = pos_id * table_dim;

  using VecHelper = VecTypeHelper<TData>;

  for (size_t i = threadIdx.x; i < table_dim; i += blockDim.x) {
    float sin_val =
        Caster<kDev>::template Cast<float>(sin_ptr[table_offset + i]);
    float cos_val =
        Caster<kDev>::template Cast<float>(cos_ptr[table_offset + i]);

    if constexpr (IsNeox) {
      if constexpr (std::is_same<TData, half>::value ||
                    std::is_same<TData, cuda_bfloat16>::value) {
        auto& y = reinterpret_cast<typename VecHelper::type2&>(
            out_ptr[out_offset + 2 * i]);
        auto& x = reinterpret_cast<const typename VecHelper::type2&>(
            input_ptr[input_offset + 2 * i]);

        float x0 = VecHelper::low(x);
        float x1 = VecHelper::high(x);

        float y0 = x0 * cos_val - x1 * sin_val;
        float y1 = x0 * sin_val + x1 * cos_val;

        y = VecHelper::make_half2(y0, y1);
      } else {
        float x0 =
            Caster<kDev>::template Cast<float>(input_ptr[input_offset + 2 * i]);
        float x1 = Caster<kDev>::template Cast<float>(
            input_ptr[input_offset + 2 * i + 1]);
        out_ptr[out_offset + 2 * i] =
            Caster<kDev>::template Cast<TData>(x0 * cos_val - x1 * sin_val);
        out_ptr[out_offset + 2 * i + 1] =
            Caster<kDev>::template Cast<TData>(x0 * sin_val + x1 * cos_val);
      }
    } else {
      size_t pos0 = i;
      size_t pos1 = i + table_dim;

      float x0 =
          Caster<kDev>::template Cast<float>(input_ptr[input_offset + pos0]);
      float x1 =
          Caster<kDev>::template Cast<float>(input_ptr[input_offset + pos1]);

      float y0 = x0 * cos_val - x1 * sin_val;
      float y1 = x0 * sin_val + x1 * cos_val;

      out_ptr[out_offset + pos0] = Caster<kDev>::template Cast<TData>(y0);
      out_ptr[out_offset + pos1] = Caster<kDev>::template Cast<TData>(y1);
    }
  }
}

}  // namespace infini::ops

#endif
