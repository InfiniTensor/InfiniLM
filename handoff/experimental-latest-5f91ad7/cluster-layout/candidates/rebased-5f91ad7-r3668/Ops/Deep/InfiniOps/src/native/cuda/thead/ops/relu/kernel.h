#ifndef INFINI_OPS_THEAD_RELU_KERNEL_H_
#define INFINI_OPS_THEAD_RELU_KERNEL_H_

#include <utility>

#include "native/cuda/ops/relu/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Relu, Device::Type::kThead>
    : public CudaRelu<Runtime<Device::Type::kThead>> {
 public:
  using CudaRelu<Runtime<Device::Type::kThead>>::CudaRelu;
};

}  // namespace infini::ops

#endif
