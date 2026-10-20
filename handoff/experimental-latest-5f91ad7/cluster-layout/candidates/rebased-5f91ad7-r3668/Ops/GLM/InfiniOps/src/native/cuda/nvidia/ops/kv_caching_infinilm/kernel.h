#ifndef INFINI_OPS_NVIDIA_KV_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_KV_CACHING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/kv_caching_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<KvCachingInfinilm, Device::Type::kNvidia>
    : public CudaKvCachingInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaKvCachingInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaKvCachingInfinilm;
};

}  // namespace infini::ops

#endif
