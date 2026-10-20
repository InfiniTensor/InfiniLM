#ifndef INFINI_OPS_MARS_KV_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_KV_CACHING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/kv_caching_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<KvCachingInfinilm, Device::Type::kMars>
    : public CudaKvCachingInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaKvCachingInfinilm<
      Runtime<Device::Type::kMars>>::CudaKvCachingInfinilm;
};

}  // namespace infini::ops

#endif
