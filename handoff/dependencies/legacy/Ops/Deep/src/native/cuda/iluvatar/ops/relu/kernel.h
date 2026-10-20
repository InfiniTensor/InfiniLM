#ifndef INFINI_OPS_ILUVATAR_RELU_KERNEL_H_
#define INFINI_OPS_ILUVATAR_RELU_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/relu/kernel.h"

namespace infini::ops {

template <>
class Operator<Relu, Device::Type::kIluvatar>
    : public CudaRelu<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaRelu<Runtime<Device::Type::kIluvatar>>::CudaRelu;
};

}  // namespace infini::ops

#endif
