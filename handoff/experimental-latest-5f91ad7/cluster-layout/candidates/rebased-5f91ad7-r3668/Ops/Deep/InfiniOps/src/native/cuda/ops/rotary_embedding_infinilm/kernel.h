#ifndef INFINI_OPS_CUDA_ROPE_KERNEL_H_
#define INFINI_OPS_CUDA_ROPE_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "base/rotary_embedding_infinilm.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/rotary_embedding_infinilm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

namespace {

template <bool IsNeox, typename Backend, typename T, typename TAngle>
void LaunchRopeKernel(dim3 grid, int block_size,
                      typename Backend::Stream cuda_stream, T* out_ptr,
                      const T* input_ptr, const void* pos_ids_ptr,
                      DataType pos_dtype, const TAngle* sin_ptr,
                      const TAngle* cos_ptr, size_t table_dim,
                      ptrdiff_t out_stride_batch, ptrdiff_t out_stride_seqlen,
                      ptrdiff_t out_stride_nhead, ptrdiff_t input_stride_batch,
                      ptrdiff_t input_stride_seqlen,
                      ptrdiff_t input_stride_nhead, ptrdiff_t pos_stride_batch,
                      bool pos_has_batch_dim, bool has_batch_dim) {
  if (pos_dtype == DataType::kInt64) {
    RopeKernel<IsNeox, Backend::kDeviceType, T, int64_t, TAngle>
        <<<grid, block_size, 0, cuda_stream>>>(
            out_ptr, input_ptr, reinterpret_cast<const int64_t*>(pos_ids_ptr),
            sin_ptr, cos_ptr, table_dim, out_stride_batch, out_stride_seqlen,
            out_stride_nhead, input_stride_batch, input_stride_seqlen,
            input_stride_nhead, pos_stride_batch, pos_has_batch_dim,
            has_batch_dim);
  } else {
    RopeKernel<IsNeox, Backend::kDeviceType, T, int32_t, TAngle>
        <<<grid, block_size, 0, cuda_stream>>>(
            out_ptr, input_ptr, reinterpret_cast<const int32_t*>(pos_ids_ptr),
            sin_ptr, cos_ptr, table_dim, out_stride_batch, out_stride_seqlen,
            out_stride_nhead, input_stride_batch, input_stride_seqlen,
            input_stride_nhead, pos_stride_batch, pos_has_batch_dim,
            has_batch_dim);
  }
}

}  // namespace

template <typename Backend>
class CudaRotaryEmbeddingInfinilm : public RotaryEmbeddingInfinilm {
 public:
  using RotaryEmbeddingInfinilm::RotaryEmbeddingInfinilm;

  void operator()(const Tensor input, const Tensor pos_ids,
                  const Tensor sin_table, const Tensor cos_table, bool is_neox,
                  Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    dim3 grid(static_cast<unsigned>(seq_len_), static_cast<unsigned>(nhead_),
              static_cast<unsigned>(batch_size_));

    assert(out.dtype() == input.dtype());
    assert(pos_ids.dtype() == DataType::kInt64 ||
           pos_ids.dtype() == DataType::kInt32);

    ptrdiff_t out_stride_batch = ndim_ == 4 ? out_strides_[0] : 0;
    ptrdiff_t out_stride_seqlen = out_strides_[ndim_ - 3];
    ptrdiff_t out_stride_nhead = out_strides_[ndim_ - 2];

    ptrdiff_t input_stride_batch = ndim_ == 4 ? input_strides_[0] : 0;
    ptrdiff_t input_stride_seqlen = input_strides_[ndim_ - 3];
    ptrdiff_t input_stride_nhead = input_strides_[ndim_ - 2];

    ptrdiff_t pos_stride_batch = pos_has_batch_dim_ ? pos_strides_[0] : 0;

    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    using DataTypes = ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>;
    using AngleTypes = ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>;

    if (is_neox) {
      DispatchFunc<DataTypes, AngleTypes, AllCudaBlockSizes>(
          {static_cast<int64_t>(out.dtype()),
           static_cast<int64_t>(sin_table.dtype()), block_size},
          [&](auto list_tag) {
            using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
            using TAngle =
                TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
            constexpr int kBlockSize = ListGet<2>(list_tag);

            LaunchRopeKernel<true, Backend, T, TAngle>(
                grid, kBlockSize, cuda_stream, reinterpret_cast<T*>(out.data()),
                reinterpret_cast<const T*>(input.data()), pos_ids.data(),
                pos_ids.dtype(),
                reinterpret_cast<const TAngle*>(sin_table.data()),
                reinterpret_cast<const TAngle*>(cos_table.data()), table_dim_,
                out_stride_batch, out_stride_seqlen, out_stride_nhead,
                input_stride_batch, input_stride_seqlen, input_stride_nhead,
                pos_stride_batch, pos_has_batch_dim_, has_batch_dim_);
          },
          "CudaRotaryEmbeddingInfinilm::operator() (Neox)");
    } else {
      DispatchFunc<DataTypes, AngleTypes, AllCudaBlockSizes>(
          {static_cast<int64_t>(out.dtype()),
           static_cast<int64_t>(sin_table.dtype()), block_size},
          [&](auto list_tag) {
            using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
            using TAngle =
                TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
            constexpr int kBlockSize = ListGet<2>(list_tag);

            LaunchRopeKernel<false, Backend, T, TAngle>(
                grid, kBlockSize, cuda_stream, reinterpret_cast<T*>(out.data()),
                reinterpret_cast<const T*>(input.data()), pos_ids.data(),
                pos_ids.dtype(),
                reinterpret_cast<const TAngle*>(sin_table.data()),
                reinterpret_cast<const TAngle*>(cos_table.data()), table_dim_,
                out_stride_batch, out_stride_seqlen, out_stride_nhead,
                input_stride_batch, input_stride_seqlen, input_stride_nhead,
                pos_stride_batch, pos_has_batch_dim_, has_batch_dim_);
          },
          "CudaRotaryEmbeddingInfinilm::operator() (Standard)");
    }
  }
};

}  // namespace infini::ops

#endif
