#ifndef INFINI_OPS_MARS_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_MARS_CONVOLUTION_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Convolution, Device::Type::kMars>
    : public CudaConv<Runtime<Device::Type::kMars>, Convolution> {
 public:
  using CudaConv<Runtime<Device::Type::kMars>, Convolution>::CudaConv;
};

}  // namespace infini::ops

#endif
