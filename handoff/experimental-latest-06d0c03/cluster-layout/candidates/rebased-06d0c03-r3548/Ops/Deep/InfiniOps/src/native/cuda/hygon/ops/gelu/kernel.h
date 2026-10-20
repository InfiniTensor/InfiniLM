#ifndef INFINI_OPS_HYGON_GELU_KERNEL_H_
#define INFINI_OPS_HYGON_GELU_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/gelu/kernel.h"

namespace infini::ops {

template <>
class Operator<Gelu, Device::Type::kHygon>
    : public CudaGelu<Runtime<Device::Type::kHygon>> {
 public:
  using CudaGelu<Runtime<Device::Type::kHygon>>::CudaGelu;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_GELU_KERNEL_H_
