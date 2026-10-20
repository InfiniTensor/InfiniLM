#ifndef INFINI_OPS_MOORE_PAGED_ATTENTION_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_PAGED_ATTENTION_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/paged_attention_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<PagedAttentionInfinilm, Device::Type::kMoore>
    : public CudaPagedAttentionInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaPagedAttentionInfinilm<
      Runtime<Device::Type::kMoore>>::CudaPagedAttentionInfinilm;
};

}  // namespace infini::ops

#endif
