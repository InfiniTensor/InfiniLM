#ifndef INFINI_OPS_ILUVATAR_ROTARY_EMBEDDING_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ROTARY_EMBEDDING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/rotary_embedding_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbeddingInfinilm, Device::Type::kIluvatar>
    : public CudaRotaryEmbeddingInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaRotaryEmbeddingInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaRotaryEmbeddingInfinilm;
};

}  // namespace infini::ops

#endif
