#ifndef INFINI_OPS_HYGON_SIGMOID_KERNEL_H_
#define INFINI_OPS_HYGON_SIGMOID_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/sigmoid/kernel.h"

namespace infini::ops {

template <>
class Operator<Sigmoid, Device::Type::kHygon>
    : public CudaSigmoid<Runtime<Device::Type::kHygon>> {
 public:
  using CudaSigmoid<Runtime<Device::Type::kHygon>>::CudaSigmoid;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_SIGMOID_KERNEL_H_
