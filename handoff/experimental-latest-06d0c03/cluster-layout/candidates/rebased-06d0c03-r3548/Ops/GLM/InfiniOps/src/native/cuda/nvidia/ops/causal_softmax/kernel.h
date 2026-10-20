#ifndef INFINI_OPS_NVIDIA_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/causal_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmax, Device::Type::kNvidia>
    : public CudaCausalSoftmax<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCausalSoftmax<Runtime<Device::Type::kNvidia>>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
