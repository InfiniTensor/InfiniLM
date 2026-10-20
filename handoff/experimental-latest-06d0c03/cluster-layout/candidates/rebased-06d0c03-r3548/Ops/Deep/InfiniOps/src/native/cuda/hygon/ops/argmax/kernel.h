#ifndef INFINI_OPS_HYGON_ARGMAX_KERNEL_H_
#define INFINI_OPS_HYGON_ARGMAX_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/argmax/kernel.h"

namespace infini::ops {

template <>
class Operator<Argmax, Device::Type::kHygon>
    : public CudaArgmax<Runtime<Device::Type::kHygon>> {
 public:
  using CudaArgmax<Runtime<Device::Type::kHygon>>::CudaArgmax;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_ARGMAX_KERNEL_H_
