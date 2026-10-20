#ifndef INFINI_OPS_NVIDIA_PAGED_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_PAGED_CACHING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/paged_caching_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedCachingInfinilm, Device::Type::kNvidia>
    : public CudaPagedCachingInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaPagedCachingInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaPagedCachingInfinilm;
};

}  // namespace infini::ops

#endif
