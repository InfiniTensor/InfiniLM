#ifndef INFINI_OPS_MOORE_COPY_KERNEL_H_
#define INFINI_OPS_MOORE_COPY_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/copy/kernel.h"

namespace infini::ops {

template <>
class Operator<Copy, Device::Type::kMoore>
    : public CudaCopy<Runtime<Device::Type::kMoore>> {
 public:
  using CudaCopy<Runtime<Device::Type::kMoore>>::CudaCopy;
};

}  // namespace infini::ops

#endif
