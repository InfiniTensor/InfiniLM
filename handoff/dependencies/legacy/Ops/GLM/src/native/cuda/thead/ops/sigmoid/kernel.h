#ifndef INFINI_OPS_THEAD_SIGMOID_KERNEL_H_
#define INFINI_OPS_THEAD_SIGMOID_KERNEL_H_

#include <utility>

#include "native/cuda/ops/sigmoid/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Sigmoid, Device::Type::kThead>
    : public CudaSigmoid<Runtime<Device::Type::kThead>> {
 public:
  using CudaSigmoid<Runtime<Device::Type::kThead>>::CudaSigmoid;
};

}  // namespace infini::ops

#endif
