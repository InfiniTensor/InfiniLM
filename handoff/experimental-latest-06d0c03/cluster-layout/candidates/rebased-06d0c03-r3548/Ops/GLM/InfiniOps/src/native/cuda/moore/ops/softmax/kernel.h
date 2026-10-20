#ifndef INFINI_OPS_MOORE_SOFTMAX_KERNEL_H_
#define INFINI_OPS_MOORE_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<Softmax, Device::Type::kMoore>
    : public CudaSoftmax<Runtime<Device::Type::kMoore>> {
 public:
  using CudaSoftmax<Runtime<Device::Type::kMoore>>::CudaSoftmax;
};

}  // namespace infini::ops

#endif
