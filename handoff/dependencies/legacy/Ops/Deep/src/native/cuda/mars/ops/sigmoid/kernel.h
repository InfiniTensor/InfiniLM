#ifndef INFINI_OPS_MARS_SIGMOID_KERNEL_H_
#define INFINI_OPS_MARS_SIGMOID_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/sigmoid/kernel.h"

namespace infini::ops {

template <>
class Operator<Sigmoid, Device::Type::kMars>
    : public CudaSigmoid<Runtime<Device::Type::kMars>> {
 public:
  using CudaSigmoid<Runtime<Device::Type::kMars>>::CudaSigmoid;
};

}  // namespace infini::ops

#endif
