#ifndef INFINI_OPS_ILUVATAR_KV_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_KV_CACHING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/kv_caching_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<KvCachingInfinilm, Device::Type::kIluvatar>
    : public CudaKvCachingInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaKvCachingInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaKvCachingInfinilm;
};

}  // namespace infini::ops

#endif
