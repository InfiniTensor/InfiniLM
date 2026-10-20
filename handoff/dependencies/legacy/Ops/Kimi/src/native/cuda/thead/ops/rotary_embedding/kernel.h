#ifndef INFINI_OPS_THEAD_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_THEAD_ROTARY_EMBEDDING_KERNEL_H_

#include "native/cuda/ops/rotary_embedding/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbedding, Device::Type::kThead>
    : public CudaRotaryEmbedding<Runtime<Device::Type::kThead>> {
 public:
  using CudaRotaryEmbedding<Runtime<Device::Type::kThead>>::CudaRotaryEmbedding;
};

}  // namespace infini::ops

#endif
