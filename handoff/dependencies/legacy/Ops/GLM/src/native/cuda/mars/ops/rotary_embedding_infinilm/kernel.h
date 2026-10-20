#ifndef INFINI_OPS_MARS_ROTARY_EMBEDDING_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_ROTARY_EMBEDDING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/rotary_embedding_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbeddingInfinilm, Device::Type::kMars>
    : public CudaRotaryEmbeddingInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaRotaryEmbeddingInfinilm<
      Runtime<Device::Type::kMars>>::CudaRotaryEmbeddingInfinilm;
};

}  // namespace infini::ops

#endif
