#ifndef INFINI_OPS_METAX_PAGED_ATTENTION_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_PAGED_ATTENTION_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/paged_attention_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionInfinilm, Device::Type::kMetax>
    : public CudaPagedAttentionInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaPagedAttentionInfinilm<
      Runtime<Device::Type::kMetax>>::CudaPagedAttentionInfinilm;
};

}  // namespace infini::ops

#endif
