#ifndef INFINI_OPS_ILUVATAR_GELU_KERNEL_H_
#define INFINI_OPS_ILUVATAR_GELU_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/gelu/kernel.h"

namespace infini::ops {

template <>
class Operator<Gelu, Device::Type::kIluvatar>
    : public CudaGelu<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaGelu<Runtime<Device::Type::kIluvatar>>::CudaGelu;
};

}  // namespace infini::ops

#endif
