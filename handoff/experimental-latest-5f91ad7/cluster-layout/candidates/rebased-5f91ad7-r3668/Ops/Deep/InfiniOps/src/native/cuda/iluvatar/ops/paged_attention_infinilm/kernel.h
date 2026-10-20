#ifndef INFINI_OPS_ILUVATAR_PAGED_ATTENTION_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_PAGED_ATTENTION_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/paged_attention_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionInfinilm, Device::Type::kIluvatar>
    : public CudaPagedAttentionInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaPagedAttentionInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaPagedAttentionInfinilm;
};

}  // namespace infini::ops

#endif
