#ifndef INFINI_OPS_MARS_RELU_KERNEL_H_
#define INFINI_OPS_MARS_RELU_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/relu/kernel.h"

namespace infini::ops {

template <>
class Operator<Relu, Device::Type::kMars>
    : public CudaRelu<Runtime<Device::Type::kMars>> {
 public:
  using CudaRelu<Runtime<Device::Type::kMars>>::CudaRelu;
};

}  // namespace infini::ops

#endif
