#ifndef INFINI_OPS_ILUVATAR_PAGED_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_PAGED_CACHING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/paged_caching_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedCachingInfinilm, Device::Type::kIluvatar>
    : public CudaPagedCachingInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaPagedCachingInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaPagedCachingInfinilm;
};

}  // namespace infini::ops

#endif
