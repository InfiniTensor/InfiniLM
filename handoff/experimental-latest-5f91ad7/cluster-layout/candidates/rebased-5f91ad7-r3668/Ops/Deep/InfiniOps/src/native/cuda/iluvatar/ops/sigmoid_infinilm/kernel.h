#ifndef INFINI_OPS_ILUVATAR_SIGMOID_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_SIGMOID_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/sigmoid_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SigmoidInfinilm, Device::Type::kIluvatar>
    : public CudaSigmoidInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaSigmoidInfinilm<
      Runtime<Device::Type::kIluvatar>>::CudaSigmoidInfinilm;
};

}  // namespace infini::ops

#endif
