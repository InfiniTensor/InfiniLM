#ifndef INFINI_OPS_MOORE_SILU_KERNEL_H_
#define INFINI_OPS_MOORE_SILU_KERNEL_H_

#include <utility>

// clang-format off
#include "native/cuda/moore/polyfills.cuh"
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/silu/kernel.h"

namespace infini::ops {

template <>
class Operator<Silu, Device::Type::kMoore>
    : public CudaSilu<Runtime<Device::Type::kMoore>> {
 public:
  using CudaSilu<Runtime<Device::Type::kMoore>>::CudaSilu;
};

}  // namespace infini::ops

#endif
