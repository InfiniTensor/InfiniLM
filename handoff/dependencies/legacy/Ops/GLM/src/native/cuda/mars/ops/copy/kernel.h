#ifndef INFINI_OPS_MARS_COPY_KERNEL_H_
#define INFINI_OPS_MARS_COPY_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/copy/kernel.h"

namespace infini::ops {

template <>
class Operator<Copy, Device::Type::kMars>
    : public CudaCopy<Runtime<Device::Type::kMars>> {
 public:
  using CudaCopy<Runtime<Device::Type::kMars>>::CudaCopy;
};

}  // namespace infini::ops

#endif
