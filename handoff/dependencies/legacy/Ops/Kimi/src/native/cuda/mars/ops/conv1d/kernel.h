#ifndef INFINI_OPS_MARS_CONV1D_KERNEL_H_
#define INFINI_OPS_MARS_CONV1D_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Conv1d, Device::Type::kMars>
    : public CudaConv<Runtime<Device::Type::kMars>, Conv1d> {
 public:
  using CudaConv<Runtime<Device::Type::kMars>, Conv1d>::CudaConv;
};

}  // namespace infini::ops

#endif
