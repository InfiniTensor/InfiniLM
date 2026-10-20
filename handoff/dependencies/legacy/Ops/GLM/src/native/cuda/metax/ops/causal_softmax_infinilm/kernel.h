#ifndef INFINI_OPS_METAX_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_METAX_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/causal_softmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmaxInfinilm, Device::Type::kMetax>
    : public CudaCausalSoftmaxInfinilm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaCausalSoftmaxInfinilm<
      Runtime<Device::Type::kMetax>>::CudaCausalSoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
