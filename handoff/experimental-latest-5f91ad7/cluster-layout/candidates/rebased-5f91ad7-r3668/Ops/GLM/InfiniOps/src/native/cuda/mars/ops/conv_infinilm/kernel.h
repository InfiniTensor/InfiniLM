#ifndef INFINI_OPS_MARS_CONV_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_CONV_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/conv_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ConvInfinilm, Device::Type::kMars>
    : public CudaConvInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaConvInfinilm<Runtime<Device::Type::kMars>>::CudaConvInfinilm;
};

}  // namespace infini::ops

#endif
