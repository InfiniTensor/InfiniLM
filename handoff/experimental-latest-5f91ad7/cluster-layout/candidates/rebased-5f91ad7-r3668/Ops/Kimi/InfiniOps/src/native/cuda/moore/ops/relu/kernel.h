#ifndef INFINI_OPS_MOORE_RELU_KERNEL_H_
#define INFINI_OPS_MOORE_RELU_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/relu/kernel.h"

namespace infini::ops {

template <>
class Operator<Relu, Device::Type::kMoore>
    : public CudaRelu<Runtime<Device::Type::kMoore>> {
 public:
  using CudaRelu<Runtime<Device::Type::kMoore>>::CudaRelu;
};

}  // namespace infini::ops

#endif
