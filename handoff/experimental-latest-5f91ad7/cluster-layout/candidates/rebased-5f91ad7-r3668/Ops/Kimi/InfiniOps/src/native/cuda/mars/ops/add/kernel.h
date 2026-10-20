#ifndef INFINI_OPS_MARS_ADD_KERNEL_H_
#define INFINI_OPS_MARS_ADD_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/add/kernel.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kMars>
    : public CudaAdd<Runtime<Device::Type::kMars>> {
 public:
  using CudaAdd<Runtime<Device::Type::kMars>>::CudaAdd;
};

}  // namespace infini::ops

#endif
