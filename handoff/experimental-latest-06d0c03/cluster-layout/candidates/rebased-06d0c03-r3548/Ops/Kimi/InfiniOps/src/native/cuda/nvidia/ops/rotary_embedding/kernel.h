#ifndef INFINI_OPS_NVIDIA_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_NVIDIA_ROTARY_EMBEDDING_KERNEL_H_

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/rotary_embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbedding, Device::Type::kNvidia>
    : public CudaRotaryEmbedding<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaRotaryEmbedding<
      Runtime<Device::Type::kNvidia>>::CudaRotaryEmbedding;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
