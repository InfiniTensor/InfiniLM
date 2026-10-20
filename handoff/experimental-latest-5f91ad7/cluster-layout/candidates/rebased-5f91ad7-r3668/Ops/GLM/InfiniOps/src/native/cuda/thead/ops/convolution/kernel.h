#ifndef INFINI_OPS_THEAD_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_THEAD_CONVOLUTION_KERNEL_H_

#include <utility>

#include "native/cuda/ops/convolution/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Convolution, Device::Type::kThead>
    : public CudaConv<Runtime<Device::Type::kThead>, Convolution> {
 public:
  using CudaConv<Runtime<Device::Type::kThead>, Convolution>::CudaConv;
};

}  // namespace infini::ops

#endif
