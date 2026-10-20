#ifndef INFINI_OPS_HYGON_COPY_KERNEL_H_
#define INFINI_OPS_HYGON_COPY_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/copy/kernel.h"

namespace infini::ops {

template <>
class Operator<Copy, Device::Type::kHygon>
    : public CudaCopy<Runtime<Device::Type::kHygon>> {
 public:
  using CudaCopy<Runtime<Device::Type::kHygon>>::CudaCopy;
};

}  // namespace infini::ops

#endif
