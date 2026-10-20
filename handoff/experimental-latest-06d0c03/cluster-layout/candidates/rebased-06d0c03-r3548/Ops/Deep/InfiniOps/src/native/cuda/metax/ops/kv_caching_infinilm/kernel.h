#ifndef INFINI_OPS_METAX_KV_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_KV_CACHING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/kv_caching_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<KvCachingInfinilm, Device::Type::kMetax>
    : public CudaKvCachingInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaKvCachingInfinilm<
      Runtime<Device::Type::kMetax>>::CudaKvCachingInfinilm;
};

}  // namespace infini::ops

#endif
