#ifndef INFINI_OPS_MOORE_FILL_KERNEL_H_
#define INFINI_OPS_MOORE_FILL_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/fill/kernel.h"

namespace infini::ops {

template <>
class Operator<Fill, Device::Type::kMoore>
    : public CudaFill<Runtime<Device::Type::kMoore>> {
 public:
  using CudaFill<Runtime<Device::Type::kMoore>>::CudaFill;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_MOORE_FILL_KERNEL_H_
