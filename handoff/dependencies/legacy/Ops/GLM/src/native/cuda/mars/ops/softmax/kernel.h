#ifndef INFINI_OPS_MARS_SOFTMAX_KERNEL_H_
#define INFINI_OPS_MARS_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<Softmax, Device::Type::kMars>
    : public CudaSoftmax<Runtime<Device::Type::kMars>> {
 public:
  using CudaSoftmax<Runtime<Device::Type::kMars>>::CudaSoftmax;
};

}  // namespace infini::ops

#endif
