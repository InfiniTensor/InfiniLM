#ifndef INFINI_OPS_MARS_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_MARS_ROTARY_EMBEDDING_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/rotary_embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbedding, Device::Type::kMars>
    : public CudaRotaryEmbedding<Runtime<Device::Type::kMars>> {
 public:
  using CudaRotaryEmbedding<Runtime<Device::Type::kMars>>::CudaRotaryEmbedding;
};

}  // namespace infini::ops

#endif
