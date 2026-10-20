#ifndef INFINI_OPS_NVIDIA_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_PAGED_ATTENTION_PREFILL_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/paged_attention_prefill_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionPrefillInfinilm, Device::Type::kNvidia>
    : public CudaPagedAttentionPrefillInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaPagedAttentionPrefillInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaPagedAttentionPrefillInfinilm;
};

}  // namespace infini::ops

#endif
