#ifndef INFINI_OPS_MOORE_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_MOORE_SILU_AND_MUL_KERNEL_H_

#include <utility>

// clang-format off
#include "native/cuda/moore/polyfills.cuh"
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/silu_and_mul/kernel.h"

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kMoore>
    : public CudaSiluAndMul<Runtime<Device::Type::kMoore>> {
 public:
  using CudaSiluAndMul<Runtime<Device::Type::kMoore>>::CudaSiluAndMul;
};

}  // namespace infini::ops

#endif
