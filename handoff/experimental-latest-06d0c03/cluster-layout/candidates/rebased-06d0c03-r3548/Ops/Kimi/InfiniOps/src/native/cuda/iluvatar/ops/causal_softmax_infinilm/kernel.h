#ifndef INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/causal_softmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmaxInfinilm, Device::Type::kIluvatar>
    : public CudaCausalSoftmaxInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaCausalSoftmaxInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaCausalSoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
