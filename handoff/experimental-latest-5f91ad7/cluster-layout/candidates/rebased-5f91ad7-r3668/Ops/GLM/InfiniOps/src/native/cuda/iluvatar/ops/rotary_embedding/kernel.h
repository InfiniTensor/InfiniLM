#ifndef INFINI_OPS_ILUVATAR_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ROTARY_EMBEDDING_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/rotary_embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbedding, Device::Type::kIluvatar>
    : public CudaRotaryEmbedding<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaRotaryEmbedding<
      Runtime<Device::Type::kIluvatar>>::CudaRotaryEmbedding;
};

}  // namespace infini::ops

#endif
