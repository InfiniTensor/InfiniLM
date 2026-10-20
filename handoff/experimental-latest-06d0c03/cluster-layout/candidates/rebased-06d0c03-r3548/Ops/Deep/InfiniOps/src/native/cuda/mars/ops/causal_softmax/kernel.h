#ifndef INFINI_OPS_MARS_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_MARS_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/causal_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmax, Device::Type::kMars>
    : public CudaCausalSoftmax<Runtime<Device::Type::kMars>> {
 public:
  using CudaCausalSoftmax<Runtime<Device::Type::kMars>>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
