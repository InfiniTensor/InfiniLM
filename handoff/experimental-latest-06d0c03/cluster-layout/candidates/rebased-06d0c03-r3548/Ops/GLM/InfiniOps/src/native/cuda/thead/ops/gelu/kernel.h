#ifndef INFINI_OPS_THEAD_GELU_KERNEL_H_
#define INFINI_OPS_THEAD_GELU_KERNEL_H_

#include <utility>

#include "native/cuda/ops/gelu/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Gelu, Device::Type::kThead>
    : public CudaGelu<Runtime<Device::Type::kThead>> {
 public:
  using CudaGelu<Runtime<Device::Type::kThead>>::CudaGelu;
};

}  // namespace infini::ops

#endif
