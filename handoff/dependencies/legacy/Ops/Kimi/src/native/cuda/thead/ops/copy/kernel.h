#ifndef INFINI_OPS_THEAD_COPY_KERNEL_H_
#define INFINI_OPS_THEAD_COPY_KERNEL_H_

#include "native/cuda/ops/copy/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Copy, Device::Type::kThead>
    : public CudaCopy<Runtime<Device::Type::kThead>> {
 public:
  using CudaCopy<Runtime<Device::Type::kThead>>::CudaCopy;
};

}  // namespace infini::ops

#endif
