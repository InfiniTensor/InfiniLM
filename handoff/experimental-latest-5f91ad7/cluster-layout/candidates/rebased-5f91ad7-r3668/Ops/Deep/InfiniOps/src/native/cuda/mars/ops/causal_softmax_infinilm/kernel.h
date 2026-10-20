#ifndef INFINI_OPS_MARS_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/causal_softmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmaxInfinilm, Device::Type::kMars>
    : public CudaCausalSoftmaxInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaCausalSoftmaxInfinilm<
      Runtime<Device::Type::kMars>>::CudaCausalSoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
