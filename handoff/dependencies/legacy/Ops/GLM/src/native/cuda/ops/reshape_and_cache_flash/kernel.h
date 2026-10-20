#ifndef INFINI_OPS_CUDA_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_CUDA_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <string>

#include "base/reshape_and_cache_flash.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/reshape_and_cache_flash/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

// Backends may lower the launch cap for this kernel without changing the
// default launch policy used by other CUDA-compatible platforms.
template <typename Backend,
          int kMaxBlockSize = BackendMaxBlockSize<Backend>::value>
class CudaReshapeAndCacheFlash : public ReshapeAndCacheFlash {
 public:
  using ReshapeAndCacheFlash::ReshapeAndCacheFlash;

  void operator()(const Tensor key, const Tensor value,
                  const Tensor slot_mapping, const Tensor /*k_scale*/,
                  const Tensor /*v_scale*/,
                  const std::string /*kv_cache_dtype*/, Tensor key_cache,
                  Tensor value_cache) const override {
    if (num_tokens_ == 0) {
      return;
    }

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    int block_size =
        std::min(RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(),
                 kMaxBlockSize);
    dim3 grid(static_cast<unsigned>(num_heads_),
              static_cast<unsigned>(num_tokens_));

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
                 SupportedCudaBlockSizesType<kMaxBlockSize>>(
        {static_cast<int64_t>(dtype_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          ReshapeAndCacheFlashKernel<T, kBlockSize>
              <<<grid, kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(key_cache.data()),
                  reinterpret_cast<T*>(value_cache.data()),
                  reinterpret_cast<const T*>(key.data()),
                  reinterpret_cast<const T*>(value.data()),
                  reinterpret_cast<const int64_t*>(slot_mapping.data()),
                  head_size_, block_size_, key_token_stride_,
                  value_token_stride_, key_head_stride_, value_head_stride_,
                  key_cache_block_stride_, value_cache_block_stride_,
                  key_cache_head_stride_, value_cache_head_stride_,
                  key_cache_page_stride_, value_cache_page_stride_);
        },
        "CudaReshapeAndCacheFlash::operator()");
  }
};

}  // namespace infini::ops

#endif
