#ifndef INFINI_OPS_MARS_FILL_KERNEL_H_
#define INFINI_OPS_MARS_FILL_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/fill/kernel.h"

namespace infini::ops {

template <>
class Operator<Fill, Device::Type::kMars>
    : public CudaFill<Runtime<Device::Type::kMars>> {
 public:
  using CudaFill<Runtime<Device::Type::kMars>>::CudaFill;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_MARS_FILL_KERNEL_H_
