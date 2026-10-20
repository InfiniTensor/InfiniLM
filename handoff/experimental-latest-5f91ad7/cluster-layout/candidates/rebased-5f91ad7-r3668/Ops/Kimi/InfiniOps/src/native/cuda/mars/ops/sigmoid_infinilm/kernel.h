#ifndef INFINI_OPS_MARS_SIGMOID_INFINILM_KERNEL_H_
#define INFINI_OPS_MARS_SIGMOID_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/sigmoid_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SigmoidInfinilm, Device::Type::kMars>
    : public CudaSigmoidInfinilm<Runtime<Device::Type::kMars>> {
 public:
  using CudaSigmoidInfinilm<Runtime<Device::Type::kMars>>::CudaSigmoidInfinilm;
};

}  // namespace infini::ops

#endif
