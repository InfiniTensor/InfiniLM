#ifndef INFINI_OPS_MOORE_MUL_KERNEL_H_
#define INFINI_OPS_MOORE_MUL_KERNEL_H_

#include <utility>

// clang-format off
#include "native/cuda/moore/polyfills.cuh"
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/mul/kernel.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kMoore>
    : public CudaMul<Runtime<Device::Type::kMoore>> {
 public:
  using CudaMul<Runtime<Device::Type::kMoore>>::CudaMul;
};

}  // namespace infini::ops

#endif
