#ifndef INFINI_OPS_ILUVATAR_SIGMOID_KERNEL_H_
#define INFINI_OPS_ILUVATAR_SIGMOID_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/sigmoid/kernel.h"

namespace infini::ops {

template <>
class Operator<Sigmoid, Device::Type::kIluvatar>
    : public CudaSigmoid<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaSigmoid<Runtime<Device::Type::kIluvatar>>::CudaSigmoid;
};

}  // namespace infini::ops

#endif
