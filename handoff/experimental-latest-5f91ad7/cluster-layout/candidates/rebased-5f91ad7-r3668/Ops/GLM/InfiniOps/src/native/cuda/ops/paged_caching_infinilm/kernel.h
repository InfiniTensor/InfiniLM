#ifndef INFINI_OPS_CUDA_PAGED_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_PAGED_CACHING_INFINILM_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "base/paged_caching_infinilm.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/reshape_and_cache_flash/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaPagedCachingInfinilm : public PagedCachingInfinilm {
 public:
  using PagedCachingInfinilm::PagedCachingInfinilm;

  void operator()(const Tensor k, const Tensor v, const Tensor slot_mapping,
                  Tensor k_cache, Tensor v_cache) const override {
    assert(k.dtype() == dtype_ && v.dtype() == dtype_ &&
           k_cache.dtype() == dtype_ && v_cache.dtype() == dtype_);
    assert(slot_mapping.dtype() == DataType::kInt64);

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    int block_size =
        std::min(RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(),
                 BackendMaxBlockSize<Backend>::value);

    dim3 grid(static_cast<unsigned>(num_kv_heads_),
              static_cast<unsigned>(num_tokens_));

    DispatchFunc<
        ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
        SupportedCudaBlockSizesType<BackendMaxBlockSize<Backend>::value>>(
        {static_cast<int64_t>(dtype_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          ReshapeAndCacheFlashKernel<T, kBlockSize>
              <<<grid, kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(k_cache.data()),
                  reinterpret_cast<T*>(v_cache.data()),
                  reinterpret_cast<const T*>(k.data()),
                  reinterpret_cast<const T*>(v.data()),
                  reinterpret_cast<const int64_t*>(slot_mapping.data()),
                  head_size_, block_size_, k_src_stride_, v_src_stride_,
                  static_cast<Tensor::Stride>(head_size_),
                  static_cast<Tensor::Stride>(head_size_),
                  k_cache_block_stride_, v_cache_block_stride_,
                  k_cache_head_stride_, v_cache_head_stride_,
                  k_cache_slot_stride_, v_cache_slot_stride_);
        },
        "CudaPagedCachingInfinilm::operator()");
  }
};

}  // namespace infini::ops

#endif
