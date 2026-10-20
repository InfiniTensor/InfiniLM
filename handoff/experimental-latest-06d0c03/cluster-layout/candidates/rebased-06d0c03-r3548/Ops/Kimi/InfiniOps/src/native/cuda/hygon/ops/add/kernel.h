#ifndef INFINI_OPS_HYGON_ADD_KERNEL_H_
#define INFINI_OPS_HYGON_ADD_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/add/kernel.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kHygon>
    : public CudaAdd<Runtime<Device::Type::kHygon>> {
 public:
  using CudaAdd<Runtime<Device::Type::kHygon>>::CudaAdd;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_ADD_KERNEL_H_
