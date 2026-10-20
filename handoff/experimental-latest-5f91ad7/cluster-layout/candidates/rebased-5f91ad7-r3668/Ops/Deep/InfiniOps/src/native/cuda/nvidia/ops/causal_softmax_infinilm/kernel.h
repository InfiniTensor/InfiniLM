#ifndef INFINI_OPS_NVIDIA_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/causal_softmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmaxInfinilm, Device::Type::kNvidia>
    : public CudaCausalSoftmaxInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaCausalSoftmaxInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaCausalSoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
