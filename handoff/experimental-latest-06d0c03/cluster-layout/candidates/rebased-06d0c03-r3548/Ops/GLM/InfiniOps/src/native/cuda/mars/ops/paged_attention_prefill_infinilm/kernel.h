#ifndef INFINI_OPS_MARS_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/paged_attention_prefill_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionPrefillInfinilm, Device::Type::kMars>
    : public CudaPagedAttentionPrefillInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaPagedAttentionPrefillInfinilm<
      Runtime<Device::Type::kMars>>::CudaPagedAttentionPrefillInfinilm;
};

}  // namespace infini::ops

#endif
