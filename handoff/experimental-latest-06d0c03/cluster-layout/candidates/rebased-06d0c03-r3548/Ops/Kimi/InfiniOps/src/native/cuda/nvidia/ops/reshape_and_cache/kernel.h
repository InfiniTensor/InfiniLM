#ifndef INFINI_OPS_NVIDIA_RESHAPE_AND_CACHE_KERNEL_H_
#define INFINI_OPS_NVIDIA_RESHAPE_AND_CACHE_KERNEL_H_

#include <cstdint>
#include <string>
#include <type_traits>

#include "base/reshape_and_cache.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/nvidia/ops/reshape_and_cache/kernel.cuh"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCache, Device::Type::kNvidia>
    : public ReshapeAndCache {
 public:
  using ReshapeAndCache::ReshapeAndCache;

  void operator()(const Tensor key, const Tensor value, Tensor key_cache,
                  Tensor value_cache, const Tensor slot_mapping,
                  const Tensor k_scale, const Tensor v_scale,
                  const std::string kv_cache_dtype) const override {
    if (num_tokens_ == 0) {
      return;
    }

    auto cuda_stream = static_cast<Runtime<Device::Type::kNvidia>::Stream>(
        stream_ ? stream_ : 0);
    const bool quantized = kv_cache_dtype != "auto";
    const bool use_e5m2 = kv_cache_dtype == "fp8_e5m2";

    DispatchFunc<Device::Type::kNvidia,
                 ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        key.dtype(),
        [&](auto type_tag) {
          using T = typename decltype(type_tag)::type;
          auto launch = [&](auto cache_tag, auto quantized_tag, auto e5m2_tag) {
            using TCache = typename decltype(cache_tag)::type;
            constexpr bool kQuantized = decltype(quantized_tag)::value;
            constexpr bool kUseE5M2 = decltype(e5m2_tag)::value;
            ReshapeAndCacheKernel<T, TCache, kQuantized, kUseE5M2>
                <<<static_cast<uint32_t>(num_tokens_), 256, 0, cuda_stream>>>(
                    reinterpret_cast<const T*>(key.data()),
                    reinterpret_cast<const T*>(value.data()),
                    reinterpret_cast<TCache*>(key_cache.data()),
                    reinterpret_cast<TCache*>(value_cache.data()),
                    reinterpret_cast<const int64_t*>(slot_mapping.data()),
                    key_strides_[0], value_strides_[0], key_strides_[1],
                    value_strides_[1], num_heads_, head_size_, block_size_, x_,
                    reinterpret_cast<const float*>(k_scale.data()),
                    reinterpret_cast<const float*>(v_scale.data()));
          };

          if (quantized) {
            if (use_e5m2) {
              launch(TypeTag<uint8_t>{}, std::true_type{}, std::true_type{});
            } else {
              launch(TypeTag<uint8_t>{}, std::true_type{}, std::false_type{});
            }
          } else {
            launch(type_tag, std::false_type{}, std::false_type{});
          }
        },
        "ReshapeAndCache::operator()");
  }
};

}  // namespace infini::ops

#endif
