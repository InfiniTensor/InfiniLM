#ifndef INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/causal_softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmax, Device::Type::kIluvatar>
    : public CudaCausalSoftmax<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaCausalSoftmax<Runtime<Device::Type::kIluvatar>>::CudaCausalSoftmax;
};

}  // namespace infini::ops

#endif
