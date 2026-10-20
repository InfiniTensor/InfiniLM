#ifndef INFINI_OPS_MARS_PAGED_ATTENTION_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_PAGED_ATTENTION_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/paged_attention_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionInfinilm, Device::Type::kMars>
    : public CudaPagedAttentionInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaPagedAttentionInfinilm<
      Runtime<Device::Type::kMars>>::CudaPagedAttentionInfinilm;
};

}  // namespace infini::ops

#endif
