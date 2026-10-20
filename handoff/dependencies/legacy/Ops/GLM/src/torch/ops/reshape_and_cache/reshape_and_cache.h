#ifndef INFINI_OPS_TORCH_RESHAPE_AND_CACHE_H_
#define INFINI_OPS_TORCH_RESHAPE_AND_CACHE_H_

#include <string>

#include "base/reshape_and_cache.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<ReshapeAndCache, kDev, 1> : public ReshapeAndCache {
 public:
  Operator(const Tensor key, const Tensor value, Tensor key_cache,
           Tensor value_cache, const Tensor slot_mapping, const Tensor k_scale,
           const Tensor v_scale, const std::string kv_cache_dtype);

  void operator()(const Tensor key, const Tensor value, Tensor key_cache,
                  Tensor value_cache, const Tensor slot_mapping,
                  const Tensor k_scale, const Tensor v_scale,
                  const std::string kv_cache_dtype) const override;
};

}  // namespace infini::ops

#endif
