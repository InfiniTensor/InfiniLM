#ifndef INFINI_OPS_CAMBRICON_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_CAMBRICON_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <string>

#include "base/reshape_and_cache_flash.h"
#include "dispatcher.h"
#include "native/cambricon/common.h"
#include "native/cambricon/data_type_.h"

namespace infini::ops {

template <typename T>
void ReshapeAndCacheFlashUnion(
    int core_per_cluster, int cluster_count, cnrtQueue_t queue, const void* key,
    const void* value, const void* slot_mapping, void* key_cache,
    void* value_cache, std::size_t num_tokens, std::size_t num_heads,
    std::size_t head_size, std::size_t block_size,
    std::ptrdiff_t key_token_stride, std::ptrdiff_t value_token_stride,
    std::ptrdiff_t key_head_stride, std::ptrdiff_t value_head_stride,
    std::ptrdiff_t key_cache_block_stride,
    std::ptrdiff_t value_cache_block_stride,
    std::ptrdiff_t key_cache_page_stride,
    std::ptrdiff_t value_cache_page_stride,
    std::ptrdiff_t key_cache_head_stride,
    std::ptrdiff_t value_cache_head_stride);

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kCambricon>
    : public ReshapeAndCacheFlash {
 public:
  using ReshapeAndCacheFlash::ReshapeAndCacheFlash;

  Operator(const Tensor key, const Tensor value, const Tensor slot_mapping,
           const Tensor k_scale, const Tensor v_scale,
           const std::string kv_cache_dtype, Tensor key_cache,
           Tensor value_cache)
      : ReshapeAndCacheFlash{key,       value,      slot_mapping,
                             k_scale,   v_scale,    kv_cache_dtype,
                             key_cache, value_cache} {
    cnrt_utils::GetLaunchConfig(key.device(), &core_per_cluster_,
                                &cluster_count_);
  }

  void operator()(const Tensor key, const Tensor value,
                  const Tensor slot_mapping, const Tensor /*k_scale*/,
                  const Tensor /*v_scale*/,
                  const std::string /*kv_cache_dtype*/, Tensor key_cache,
                  Tensor value_cache) const override {
    if (num_tokens_ == 0) {
      return;
    }

    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    DispatchFunc<
        Device::Type::kCambricon,
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>>(
        {dtype_},
        [&](auto dtype_tag) {
          using T = typename decltype(dtype_tag)::type;
          ReshapeAndCacheFlashUnion<T>(
              core_per_cluster_, cluster_count_, queue, key.data(),
              value.data(), slot_mapping.data(), key_cache.data(),
              value_cache.data(), num_tokens_, num_heads_, head_size_,
              block_size_, key_token_stride_, value_token_stride_,
              key_head_stride_, value_head_stride_, key_cache_block_stride_,
              value_cache_block_stride_, key_cache_page_stride_,
              value_cache_page_stride_, key_cache_head_stride_,
              value_cache_head_stride_);
        },
        "CambriconReshapeAndCacheFlash::operator()");
  }

 private:
  int core_per_cluster_{0};
  int cluster_count_{0};
};

}  // namespace infini::ops

#endif
