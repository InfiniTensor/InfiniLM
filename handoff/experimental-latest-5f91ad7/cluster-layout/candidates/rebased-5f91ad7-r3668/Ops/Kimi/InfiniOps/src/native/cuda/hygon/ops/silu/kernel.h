#ifndef INFINI_OPS_HYGON_SILU_KERNEL_H_
#define INFINI_OPS_HYGON_SILU_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/silu/kernel.h"

namespace infini::ops {

template <>
class Operator<Silu, Device::Type::kHygon>
    : public CudaSilu<Runtime<Device::Type::kHygon>> {
 public:
  using CudaSilu<Runtime<Device::Type::kHygon>>::CudaSilu;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_SILU_KERNEL_H_
