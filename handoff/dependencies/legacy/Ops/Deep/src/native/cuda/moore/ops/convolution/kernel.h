#ifndef INFINI_OPS_MOORE_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_MOORE_CONVOLUTION_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Convolution, Device::Type::kMoore>
    : public CudaConv<Runtime<Device::Type::kMoore>, Convolution> {
 public:
  using CudaConv<Runtime<Device::Type::kMoore>, Convolution>::CudaConv;
};

}  // namespace infini::ops

#endif
