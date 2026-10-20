#ifndef INFINI_OPS_HYGON_RELU_KERNEL_H_
#define INFINI_OPS_HYGON_RELU_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/relu/kernel.h"

namespace infini::ops {

template <>
class Operator<Relu, Device::Type::kHygon>
    : public CudaRelu<Runtime<Device::Type::kHygon>> {
 public:
  using CudaRelu<Runtime<Device::Type::kHygon>>::CudaRelu;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_RELU_KERNEL_H_
