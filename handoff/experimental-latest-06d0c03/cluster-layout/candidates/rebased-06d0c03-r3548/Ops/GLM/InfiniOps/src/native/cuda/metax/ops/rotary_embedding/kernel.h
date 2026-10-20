#ifndef INFINI_OPS_METAX_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_METAX_ROTARY_EMBEDDING_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/rotary_embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbedding, Device::Type::kMetax>
    : public CudaRotaryEmbedding<Runtime<Device::Type::kMetax>> {
 public:
  using CudaRotaryEmbedding<Runtime<Device::Type::kMetax>>::CudaRotaryEmbedding;
};

}  // namespace infini::ops

#endif
