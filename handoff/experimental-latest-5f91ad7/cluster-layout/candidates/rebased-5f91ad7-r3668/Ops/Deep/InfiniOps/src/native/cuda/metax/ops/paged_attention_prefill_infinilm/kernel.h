#ifndef INFINI_OPS_METAX_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/paged_attention_prefill_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionPrefillInfinilm, Device::Type::kMetax>
    : public CudaPagedAttentionPrefillInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaPagedAttentionPrefillInfinilm<
      Runtime<Device::Type::kMetax>>::CudaPagedAttentionPrefillInfinilm;
};

}  // namespace infini::ops

#endif
