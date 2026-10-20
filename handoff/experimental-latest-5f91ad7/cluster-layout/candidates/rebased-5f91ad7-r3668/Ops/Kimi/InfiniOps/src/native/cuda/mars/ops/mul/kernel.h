#ifndef INFINI_OPS_MARS_MUL_KERNEL_H_
#define INFINI_OPS_MARS_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/mul/kernel.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kMars>
    : public CudaMul<Runtime<Device::Type::kMars>> {
 public:
  using CudaMul<Runtime<Device::Type::kMars>>::CudaMul;
};

}  // namespace infini::ops

#endif
