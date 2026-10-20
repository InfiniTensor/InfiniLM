#ifndef INFINI_OPS_ILUVATAR_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_SOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/softmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SoftmaxInfinilm, Device::Type::kIluvatar>
    : public CudaSoftmaxInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaSoftmaxInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaSoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
