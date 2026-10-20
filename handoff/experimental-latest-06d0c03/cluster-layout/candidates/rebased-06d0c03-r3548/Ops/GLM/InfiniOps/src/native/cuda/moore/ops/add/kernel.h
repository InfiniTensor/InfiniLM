#ifndef INFINI_OPS_MOORE_ADD_KERNEL_H_
#define INFINI_OPS_MOORE_ADD_KERNEL_H_

#include <utility>

// clang-format off
#include "native/cuda/moore/polyfills.cuh"
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/add/kernel.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kMoore>
    : public CudaAdd<Runtime<Device::Type::kMoore>> {
 public:
  using CudaAdd<Runtime<Device::Type::kMoore>>::CudaAdd;
};

}  // namespace infini::ops

#endif
