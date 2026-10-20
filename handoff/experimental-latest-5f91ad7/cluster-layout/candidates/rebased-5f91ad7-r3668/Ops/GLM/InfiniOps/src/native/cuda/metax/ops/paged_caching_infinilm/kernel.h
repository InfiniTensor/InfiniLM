#ifndef INFINI_OPS_METAX_PAGED_CACHING_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_PAGED_CACHING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/paged_caching_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedCachingInfinilm, Device::Type::kMetax>
    : public CudaPagedCachingInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaPagedCachingInfinilm<
      Runtime<Device::Type::kMetax>>::CudaPagedCachingInfinilm;
};

}  // namespace infini::ops

#endif
