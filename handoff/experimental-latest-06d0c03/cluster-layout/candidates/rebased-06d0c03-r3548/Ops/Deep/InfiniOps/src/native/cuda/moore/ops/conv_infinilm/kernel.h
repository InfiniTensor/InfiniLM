#ifndef INFINI_OPS_MOORE_CONV_INFINILM_KERNEL_H_
#define INFINI_OPS_MOORE_CONV_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/conv_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ConvInfinilm, Device::Type::kMoore>
    : public CudaConvInfinilm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaConvInfinilm<Runtime<Device::Type::kMoore>>::CudaConvInfinilm;
};

}  // namespace infini::ops

#endif
