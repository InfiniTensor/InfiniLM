#ifndef INFINI_OPS_NVIDIA_ROTARY_EMBEDDING_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_ROTARY_EMBEDDING_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/rotary_embedding_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbeddingInfinilm, Device::Type::kNvidia>
    : public CudaRotaryEmbeddingInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaRotaryEmbeddingInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaRotaryEmbeddingInfinilm;
};

}  // namespace infini::ops

#endif
