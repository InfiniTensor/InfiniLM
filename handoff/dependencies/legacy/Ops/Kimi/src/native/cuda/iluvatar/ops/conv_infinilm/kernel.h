#ifndef INFINI_OPS_ILUVATAR_CONV_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CONV_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/conv_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ConvInfinilm, Device::Type::kIluvatar>
    : public CudaConvInfinilm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaConvInfinilm<Runtime<Device::Type::kIluvatar>>::CudaConvInfinilm;
};

}  // namespace infini::ops

#endif
