#ifndef INFINI_OPS_HYGON_SOFTMAX_KERNEL_H_
#define INFINI_OPS_HYGON_SOFTMAX_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<Softmax, Device::Type::kHygon>
    : public CudaSoftmax<Runtime<Device::Type::kHygon>> {
 public:
  using CudaSoftmax<Runtime<Device::Type::kHygon>>::CudaSoftmax;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_SOFTMAX_KERNEL_H_
