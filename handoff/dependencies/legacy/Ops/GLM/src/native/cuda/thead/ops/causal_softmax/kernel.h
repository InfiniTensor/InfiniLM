#ifndef INFINI_OPS_THEAD_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_THEAD_CAUSAL_SOFTMAX_KERNEL_H_

#include "native/cuda/ops/causal_softmax/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmax, Device::Type::kThead>
    : public CudaCausalSoftmax<Runtime<Device::Type::kThead>> {
 public:
  using CudaCausalSoftmax<Runtime<Device::Type::kThead>>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
