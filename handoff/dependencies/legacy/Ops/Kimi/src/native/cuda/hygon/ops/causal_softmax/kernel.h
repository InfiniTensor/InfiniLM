#ifndef INFINI_OPS_HYGON_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_HYGON_CAUSAL_SOFTMAX_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/causal_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmax, Device::Type::kHygon>
    : public CudaCausalSoftmax<Runtime<Device::Type::kHygon>> {
 public:
  using CudaCausalSoftmax<Runtime<Device::Type::kHygon>>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
