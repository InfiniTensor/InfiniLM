#ifndef INFINI_OPS_ILUVATAR_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/paged_attention_prefill_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionPrefillInfinilm, Device::Type::kIluvatar>
    : public CudaPagedAttentionPrefillInfinilm<
          Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaPagedAttentionPrefillInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaPagedAttentionPrefillInfinilm;
};

}  // namespace infini::ops

#endif
