#ifndef INFINI_OPS_MOORE_KV_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_KV_CACHING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/kv_caching_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<KvCachingInfinilm, Device::Type::kMoore>
    : public CudaKvCachingInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaKvCachingInfinilm<
      Runtime<Device::Type::kMoore>>::CudaKvCachingInfinilm;
};

}  // namespace infini::ops

#endif
