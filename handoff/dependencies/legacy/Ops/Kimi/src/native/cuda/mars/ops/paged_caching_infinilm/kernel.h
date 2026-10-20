#ifndef INFINI_OPS_MARS_PAGED_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_PAGED_CACHING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/paged_caching_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedCachingInfinilm, Device::Type::kMars>
    : public CudaPagedCachingInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaPagedCachingInfinilm<
      Runtime<Device::Type::kMars>>::CudaPagedCachingInfinilm;
};

}  // namespace infini::ops

#endif
