#ifndef INFINI_OPS_MOORE_SIGMOID_KERNEL_H_
#define INFINI_OPS_MOORE_SIGMOID_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/sigmoid/kernel.h"

namespace infini::ops {

template <>
class Operator<Sigmoid, Device::Type::kMoore>
    : public CudaSigmoid<Runtime<Device::Type::kMoore>> {
 public:
  using CudaSigmoid<Runtime<Device::Type::kMoore>>::CudaSigmoid;
};

}  // namespace infini::ops

#endif
