#ifndef INFINI_OPS_THEAD_SOFTMAX_KERNEL_H_
#define INFINI_OPS_THEAD_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/ops/softmax/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Softmax, Device::Type::kThead>
    : public CudaSoftmax<Runtime<Device::Type::kThead>> {
 public:
  using CudaSoftmax<Runtime<Device::Type::kThead>>::CudaSoftmax;
};

}  // namespace infini::ops

#endif
